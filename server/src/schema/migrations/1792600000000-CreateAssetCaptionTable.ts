import { Kysely, sql } from 'kysely';

export async function up(db: Kysely<any>): Promise<void> {
  await sql`CREATE OR REPLACE FUNCTION asset_caption_delete_audit()
  RETURNS TRIGGER
  LANGUAGE PLPGSQL
  AS $$
    BEGIN
      INSERT INTO asset_caption_audit ("assetId")
      SELECT "assetId"
      FROM OLD;
      RETURN NULL;
    END
  $$;`.execute(db);
  await sql`CREATE TABLE "asset_caption_audit" (
  "id" uuid NOT NULL DEFAULT immich_uuid_v7(),
  "assetId" uuid NOT NULL,
  "deletedAt" timestamp with time zone NOT NULL DEFAULT clock_timestamp(),
  CONSTRAINT "asset_caption_audit_pkey" PRIMARY KEY ("id")
);`.execute(db);
  await sql`CREATE INDEX "asset_caption_audit_assetId_idx" ON "asset_caption_audit" ("assetId");`.execute(db);
  await sql`CREATE INDEX "asset_caption_audit_deletedAt_idx" ON "asset_caption_audit" ("deletedAt");`.execute(db);
  await sql`CREATE TABLE "asset_caption" (
  "assetId" uuid NOT NULL,
  "text" text NOT NULL,
  "searchText" text NOT NULL DEFAULT '',
  "model" character varying NOT NULL,
  "updatedAt" timestamp with time zone NOT NULL DEFAULT now(),
  "updateId" uuid NOT NULL DEFAULT immich_uuid_v7(),
  CONSTRAINT "asset_caption_pkey" PRIMARY KEY ("assetId")
);`.execute(db);
  await sql`ALTER TABLE "asset_caption" ADD CONSTRAINT "asset_caption_assetId_fkey" FOREIGN KEY ("assetId") REFERENCES "asset" ("id") ON UPDATE CASCADE ON DELETE CASCADE;`.execute(
    db,
  );
  await sql`CREATE INDEX "asset_caption_updateId_idx" ON "asset_caption" ("updateId");`.execute(db);
  await sql`CREATE INDEX "idx_asset_caption_search_text" ON "asset_caption" USING gin (f_unaccent("searchText") gin_trgm_ops);`.execute(
    db,
  );
  await sql`CREATE OR REPLACE TRIGGER "asset_caption_delete_audit"
  AFTER DELETE ON "asset_caption"
  REFERENCING OLD TABLE AS "old"
  FOR EACH STATEMENT
  WHEN (pg_trigger_depth() = 0)
  EXECUTE FUNCTION asset_caption_delete_audit();`.execute(db);
  await sql`CREATE OR REPLACE TRIGGER "asset_caption_updatedAt"
  BEFORE UPDATE ON "asset_caption"
  FOR EACH ROW
  EXECUTE FUNCTION updated_at();`.execute(db);
  await sql`INSERT INTO "migration_overrides" ("name", "value") VALUES ('function_asset_caption_delete_audit', '{"type":"function","name":"asset_caption_delete_audit","sql":"CREATE OR REPLACE FUNCTION asset_caption_delete_audit()\\n  RETURNS TRIGGER\\n  LANGUAGE PLPGSQL\\n  AS $$\\n    BEGIN\\n      INSERT INTO asset_caption_audit (\\"assetId\\")\\n      SELECT \\"assetId\\"\\n      FROM OLD;\\n      RETURN NULL;\\n    END\\n  $$;"}'::jsonb);`.execute(
    db,
  );
  await sql`INSERT INTO "migration_overrides" ("name", "value") VALUES ('index_idx_asset_caption_search_text', '{"type":"index","name":"idx_asset_caption_search_text","sql":"CREATE INDEX \\"idx_asset_caption_search_text\\" ON \\"asset_caption\\" USING gin (f_unaccent(\\"searchText\\") gin_trgm_ops);"}'::jsonb);`.execute(
    db,
  );
  await sql`INSERT INTO "migration_overrides" ("name", "value") VALUES ('trigger_asset_caption_delete_audit', '{"type":"trigger","name":"asset_caption_delete_audit","sql":"CREATE OR REPLACE TRIGGER \\"asset_caption_delete_audit\\"\\n  AFTER DELETE ON \\"asset_caption\\"\\n  REFERENCING OLD TABLE AS \\"old\\"\\n  FOR EACH STATEMENT\\n  WHEN (pg_trigger_depth() = 0)\\n  EXECUTE FUNCTION asset_caption_delete_audit();"}'::jsonb);`.execute(
    db,
  );
  await sql`INSERT INTO "migration_overrides" ("name", "value") VALUES ('trigger_asset_caption_updatedAt', '{"type":"trigger","name":"asset_caption_updatedAt","sql":"CREATE OR REPLACE TRIGGER \\"asset_caption_updatedAt\\"\\n  BEFORE UPDATE ON \\"asset_caption\\"\\n  FOR EACH ROW\\n  EXECUTE FUNCTION updated_at();"}'::jsonb);`.execute(
    db,
  );
}

export async function down(db: Kysely<any>): Promise<void> {
  await sql`DROP TRIGGER "asset_caption_updatedAt" ON "asset_caption";`.execute(db);
  await sql`DROP TRIGGER "asset_caption_delete_audit" ON "asset_caption";`.execute(db);
  await sql`DROP TABLE "asset_caption";`.execute(db);
  await sql`DELETE FROM "migration_overrides" WHERE "name" = 'index_idx_asset_caption_search_text';`.execute(db);
  await sql`DROP FUNCTION asset_caption_delete_audit;`.execute(db);
  await sql`DELETE FROM "migration_overrides" WHERE "name" = 'function_asset_caption_delete_audit';`.execute(db);
  await sql`DELETE FROM "migration_overrides" WHERE "name" = 'trigger_asset_caption_delete_audit';`.execute(db);
  await sql`DELETE FROM "migration_overrides" WHERE "name" = 'trigger_asset_caption_updatedAt';`.execute(db);
}
