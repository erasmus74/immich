import { Injectable } from '@nestjs/common';
import { type Insertable, type Kysely } from 'kysely';
import { InjectKysely } from 'nestjs-kysely';
import { DummyValue, GenerateSql } from 'src/decorators.js';
import { DB } from 'src/schema/index.js';
import { AssetCaptionTable } from 'src/schema/tables/asset-caption.table.js';

@Injectable()
export class CaptionRepository {
  constructor(@InjectKysely() private db: Kysely<DB>) {}

  @GenerateSql({ params: [DummyValue.UUID] })
  getByAssetId(assetId: string) {
    return this.db
      .selectFrom('asset_caption')
      .selectAll('asset_caption')
      .where('asset_caption.assetId', '=', assetId)
      .executeTakeFirst();
  }

  deleteAll() {
    return this.db.deleteFrom('asset_caption').execute();
  }

  @GenerateSql({
    params: [DummyValue.UUID, { text: DummyValue.STRING, searchText: DummyValue.STRING, model: DummyValue.STRING }],
  })
  upsert(
    assetId: string,
    { text, searchText, model }: Pick<Insertable<AssetCaptionTable>, 'text' | 'searchText' | 'model'>,
  ) {
    return this.db
      .insertInto('asset_caption')
      .values({ assetId, text, searchText, model })
      .onConflict((oc) => oc.column('assetId').doUpdateSet({ text, searchText, model, updatedAt: new Date() }))
      .execute();
  }
}
