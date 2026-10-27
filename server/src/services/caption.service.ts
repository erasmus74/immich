import { Injectable } from '@nestjs/common';

import type { JobOf } from 'src/types.js';
import { OnJob } from 'src/decorators.js';
import { AssetVisibility, JobName, JobStatus, QueueName } from 'src/enum.js';
import { BaseService } from 'src/services/base.service.js';
import { tokenizeForSearch } from 'src/utils/database.js';
import { batched, isCaptioningEnabled } from 'src/utils/misc.js';

@Injectable()
export class CaptionService extends BaseService {
  @OnJob({ name: JobName.CaptionQueueAll, queue: QueueName.Captioning })
  async handleQueueCaptions({ force }: JobOf<JobName.CaptionQueueAll>): Promise<JobStatus> {
    const { machineLearning } = await this.getConfig({ withCache: false });
    if (!isCaptioningEnabled(machineLearning)) {
      return JobStatus.Skipped;
    }

    if (force) {
      await this.captionRepository.deleteAll();
    }

    for await (const assets of batched(this.assetJobRepository.streamForCaptionJob(force))) {
      await this.jobRepository.queueAll(assets.map((asset) => ({ name: JobName.Caption, data: { id: asset.id } })));
    }

    return JobStatus.Success;
  }

  @OnJob({ name: JobName.Caption, queue: QueueName.Captioning })
  async handleCaption({ id }: JobOf<JobName.Caption>): Promise<JobStatus> {
    const { machineLearning } = await this.getConfig({ withCache: true });
    if (!isCaptioningEnabled(machineLearning)) {
      return JobStatus.Skipped;
    }

    const asset = await this.assetJobRepository.getForCaption(id);
    if (!asset || !asset.previewFile) {
      return JobStatus.Failed;
    }

    if (asset.visibility === AssetVisibility.Hidden) {
      return JobStatus.Skipped;
    }

    const { captioning } = machineLearning;
    let modelName = captioning.modelName;
    let text: string;
    try {
      text = await this.machineLearningRepository.caption(asset.previewFile, {
        modelName,
        prompt: captioning.prompt,
        maxTokens: captioning.maxTokens,
      });
    } catch (error: Error | unknown) {
      if (!captioning.fallbackModel) {
        throw error;
      }

      this.logger.warn(
        `Unable to generate caption with ${captioning.modelName}, retrying with fallback model ${captioning.fallbackModel}`,
        error,
      );
      modelName = captioning.fallbackModel;
      text = await this.machineLearningRepository.caption(asset.previewFile, {
        modelName,
        prompt: captioning.prompt,
        maxTokens: captioning.maxTokens,
      });
    }

    await this.captionRepository.upsert(id, { text, searchText: tokenizeForSearch(text).join(' '), model: modelName });

    await this.assetRepository.upsertJobStatus({ assetId: id, captionAt: new Date() });

    this.logger.debug(`Generated caption for ${id} using ${modelName}`);
    return JobStatus.Success;
  }
}
