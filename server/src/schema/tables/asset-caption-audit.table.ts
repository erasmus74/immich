import { Column, CreateDateColumn, type Generated, Table } from '@immich/sql-tools';
import { PrimaryGeneratedUuidV7Column } from 'src/decorators.js';

@Table('asset_caption_audit')
export class AssetCaptionAuditTable {
  @PrimaryGeneratedUuidV7Column()
  id!: Generated<string>;

  @Column({ type: 'uuid', index: true })
  assetId!: string;

  @CreateDateColumn({ default: () => 'clock_timestamp()', index: true })
  deletedAt!: Date;
}
