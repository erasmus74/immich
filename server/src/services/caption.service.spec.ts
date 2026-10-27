import { AssetFileType, AssetVisibility, ImmichWorker, JobName, JobStatus } from 'src/enum.js';
import { CaptionService } from 'src/services/caption.service.js';
import { AssetFactory } from 'test/factories/asset.factory.js';
import { systemConfigStub } from 'test/fixtures/system-config.stub.js';
import { ServiceMocks, makeStream, newTestService } from 'test/utils.js';

describe(CaptionService.name, () => {
  let sut: CaptionService;
  let mocks: ServiceMocks;

  beforeEach(() => {
    ({ sut, mocks } = newTestService(CaptionService));

    mocks.config.getWorker.mockReturnValue(ImmichWorker.Microservices);
    mocks.assetJob.getForCaption.mockResolvedValue({
      visibility: AssetVisibility.Timeline,
      previewFile: '/uploads/user-id/thumbs/path.jpg',
    });
    mocks.machineLearning.caption.mockResolvedValue('A photo of a dog');
  });

  it('should work', () => {
    expect(sut).toBeDefined();
  });

  describe('handleQueueCaptions', () => {
    it('should do nothing if machine learning is disabled', async () => {
      mocks.systemMetadata.get.mockResolvedValue(systemConfigStub.machineLearningDisabled);

      await sut.handleQueueCaptions({ force: false });

      expect(mocks.database.setDimensionSize).not.toHaveBeenCalled();
    });

    it('should queue the assets without captions', async () => {
      const asset = AssetFactory.create();
      mocks.assetJob.streamForCaptionJob.mockReturnValue(makeStream([asset]));

      await sut.handleQueueCaptions({ force: false });

      expect(mocks.job.queueAll).toHaveBeenCalledWith([{ name: JobName.Caption, data: { id: asset.id } }]);
      expect(mocks.assetJob.streamForCaptionJob).toHaveBeenCalledWith(false);
    });

    it('should queue all the assets', async () => {
      const asset = AssetFactory.create();
      mocks.assetJob.streamForCaptionJob.mockReturnValue(makeStream([asset]));

      await sut.handleQueueCaptions({ force: true });

      expect(mocks.caption.deleteAll).toHaveBeenCalled();
      expect(mocks.job.queueAll).toHaveBeenCalledWith([{ name: JobName.Caption, data: { id: asset.id } }]);
      expect(mocks.assetJob.streamForCaptionJob).toHaveBeenCalledWith(true);
    });
  });

  describe('handleCaption', () => {
    it('should do nothing if machine learning is disabled', async () => {
      mocks.systemMetadata.get.mockResolvedValue(systemConfigStub.machineLearningDisabled);

      expect(await sut.handleCaption({ id: '123' })).toEqual(JobStatus.Skipped);

      expect(mocks.machineLearning.caption).not.toHaveBeenCalled();
    });

    it('should skip assets without a preview path', async () => {
      const asset = AssetFactory.create();
      mocks.assetJob.getForCaption.mockResolvedValue({ visibility: AssetVisibility.Timeline, previewFile: null });

      expect(await sut.handleCaption({ id: asset.id })).toEqual(JobStatus.Failed);

      expect(mocks.caption.upsert).not.toHaveBeenCalled();
      expect(mocks.machineLearning.caption).not.toHaveBeenCalled();
    });

    it('should save the returned caption', async () => {
      const asset = AssetFactory.create();

      expect(await sut.handleCaption({ id: asset.id })).toEqual(JobStatus.Success);

      expect(mocks.machineLearning.caption).toHaveBeenCalledWith(
        '/uploads/user-id/thumbs/path.jpg',
        expect.objectContaining({
          modelName: 'Qwen/Qwen3-VL-2B-Instruct',
          prompt: '',
          maxTokens: 120,
        }),
      );
      expect(mocks.caption.upsert).toHaveBeenCalledWith(asset.id, {
        text: 'A photo of a dog',
        searchText: 'A photo of a dog',
        model: 'Qwen/Qwen3-VL-2B-Instruct',
      });
      expect(mocks.asset.upsertJobStatus).toHaveBeenCalledWith({ assetId: asset.id, captionAt: expect.any(Date) });
    });

    it('should apply config settings', async () => {
      const asset = AssetFactory.create();
      mocks.systemMetadata.get.mockResolvedValue({
        machineLearning: {
          enabled: true,
          captioning: {
            modelName: 'microsoft/Florence-2-large',
            prompt: 'Describe this image in detail',
            maxTokens: 256,
          },
        },
      });
      mocks.machineLearning.caption.mockResolvedValue('A detailed description');

      expect(await sut.handleCaption({ id: asset.id })).toEqual(JobStatus.Success);

      expect(mocks.machineLearning.caption).toHaveBeenCalledWith(
        '/uploads/user-id/thumbs/path.jpg',
        expect.objectContaining({
          modelName: 'microsoft/Florence-2-large',
          prompt: 'Describe this image in detail',
          maxTokens: 256,
        }),
      );
      expect(mocks.caption.upsert).toHaveBeenCalledWith(asset.id, {
        text: 'A detailed description',
        searchText: 'A detailed description',
        model: 'microsoft/Florence-2-large',
      });
    });

    it('should skip invisible assets', async () => {
      const asset = AssetFactory.from().file({ type: AssetFileType.Preview }).build();
      mocks.assetJob.getForCaption.mockResolvedValue({
        visibility: AssetVisibility.Hidden,
        previewFile: asset.files[0].path,
      });

      expect(await sut.handleCaption({ id: asset.id })).toEqual(JobStatus.Skipped);

      expect(mocks.machineLearning.caption).not.toHaveBeenCalled();
      expect(mocks.caption.upsert).not.toHaveBeenCalled();
    });

    it('should fail if asset could not be found', async () => {
      mocks.assetJob.getForCaption.mockResolvedValue(void 0);

      expect(await sut.handleCaption({ id: 'non-existent' })).toEqual(JobStatus.Failed);

      expect(mocks.machineLearning.caption).not.toHaveBeenCalled();
      expect(mocks.caption.upsert).not.toHaveBeenCalled();
    });

    it('should throw if captioning fails without a fallback model', async () => {
      const asset = AssetFactory.create();
      mocks.machineLearning.caption.mockRejectedValue(new Error('caption failed'));

      await expect(sut.handleCaption({ id: asset.id })).rejects.toThrow('caption failed');

      expect(mocks.machineLearning.caption).toHaveBeenCalledTimes(1);
      expect(mocks.caption.upsert).not.toHaveBeenCalled();
      expect(mocks.asset.upsertJobStatus).not.toHaveBeenCalled();
    });

    describe('fallback model', () => {
      beforeEach(() => {
        mocks.systemMetadata.get.mockResolvedValue({
          machineLearning: {
            enabled: true,
            captioning: {
              fallbackModel: 'microsoft/Florence-2-large',
            },
          },
        });
      });

      it('should retry with the fallback model and store the fallback model name', async () => {
        const asset = AssetFactory.create();
        mocks.machineLearning.caption
          .mockRejectedValueOnce(new Error('caption failed'))
          .mockResolvedValueOnce('A fallback caption');

        expect(await sut.handleCaption({ id: asset.id })).toEqual(JobStatus.Success);

        expect(mocks.machineLearning.caption).toHaveBeenCalledTimes(2);
        expect(mocks.machineLearning.caption).toHaveBeenNthCalledWith(
          1,
          '/uploads/user-id/thumbs/path.jpg',
          expect.objectContaining({ modelName: 'Qwen/Qwen3-VL-2B-Instruct' }),
        );
        expect(mocks.machineLearning.caption).toHaveBeenNthCalledWith(
          2,
          '/uploads/user-id/thumbs/path.jpg',
          expect.objectContaining({ modelName: 'microsoft/Florence-2-large' }),
        );
        expect(mocks.caption.upsert).toHaveBeenCalledWith(asset.id, {
          text: 'A fallback caption',
          searchText: 'A fallback caption',
          model: 'microsoft/Florence-2-large',
        });
        expect(mocks.asset.upsertJobStatus).toHaveBeenCalledWith({ assetId: asset.id, captionAt: expect.any(Date) });
      });

      it('should throw if both the primary and fallback models fail', async () => {
        const asset = AssetFactory.create();
        mocks.machineLearning.caption.mockRejectedValue(new Error('caption failed'));

        await expect(sut.handleCaption({ id: asset.id })).rejects.toThrow('caption failed');

        expect(mocks.machineLearning.caption).toHaveBeenCalledTimes(2);
        expect(mocks.caption.upsert).not.toHaveBeenCalled();
        expect(mocks.asset.upsertJobStatus).not.toHaveBeenCalled();
      });
    });

    describe('search tokenization', () => {
      it('should generate bigrams for Chinese text', async () => {
        const asset = AssetFactory.create();
        mocks.machineLearning.caption.mockResolvedValue('機器學習');

        await sut.handleCaption({ id: asset.id });

        expect(mocks.caption.upsert).toHaveBeenCalledWith(
          asset.id,
          expect.objectContaining({ searchText: '機器 器學 學習' }),
        );
      });

      it('should pass through Latin text unchanged', async () => {
        const asset = AssetFactory.create();
        mocks.machineLearning.caption.mockResolvedValue('Hello World');

        await sut.handleCaption({ id: asset.id });

        expect(mocks.caption.upsert).toHaveBeenCalledWith(
          asset.id,
          expect.objectContaining({ searchText: 'Hello World' }),
        );
      });

      it('should normalize whitespace', async () => {
        const asset = AssetFactory.create();
        mocks.machineLearning.caption.mockResolvedValue('  Hello   World  ');

        await sut.handleCaption({ id: asset.id });

        expect(mocks.caption.upsert).toHaveBeenCalledWith(
          asset.id,
          expect.objectContaining({ searchText: 'Hello World' }),
        );
      });
    });
  });
});
