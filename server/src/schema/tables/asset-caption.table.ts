import {
  AfterDeleteTrigger,
  Column,
  ForeignKeyColumn,
  type Generated,
  Index,
  Table,
  Timestamp,
  UpdateDateColumn,
} from '@immich/sql-tools';
import { UpdateIdColumn, UpdatedAtTrigger } from 'src/decorators.js';
import { asset_caption_delete_audit } from 'src/schema/functions.js';
import { AssetTable } from 'src/schema/tables/asset.table.js';

@Table('asset_caption')
@Index({
  name: 'idx_asset_caption_search_text',
  using: 'gin',
  expression: 'f_unaccent("searchText") gin_trgm_ops',
})
@UpdatedAtTrigger('asset_caption_updatedAt')
@AfterDeleteTrigger({
  scope: 'statement',
  function: asset_caption_delete_audit,
  referencingOldTableAs: 'old',
  when: 'pg_trigger_depth() = 0',
})
export class AssetCaptionTable {
  @ForeignKeyColumn(() => AssetTable, { onDelete: 'CASCADE', onUpdate: 'CASCADE', primary: true })
  assetId!: string;

  @Column({ type: 'text' })
  text!: string;

  @Column({ type: 'text', default: '' })
  searchText!: string;

  @Column({ type: 'character varying' })
  model!: string;

  @UpdateDateColumn()
  updatedAt!: Generated<Timestamp>;

  @UpdateIdColumn({ index: true })
  updateId!: Generated<string>;
}
