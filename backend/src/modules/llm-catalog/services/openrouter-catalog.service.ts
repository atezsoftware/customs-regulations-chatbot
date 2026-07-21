import {createHash} from 'node:crypto';
import {PostgresDataSource} from '../../../datasources';

type OpenRouterModel = Record<string, unknown>;
type SyncResult = {received: number; active: number; skipped: number; completedAt: string};

const PROVIDER = 'openrouter';

function asArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function text(value: unknown): string | undefined { return typeof value === 'string' && value ? value : undefined; }
function numeric(value: unknown): string | null { return typeof value === 'string' || typeof value === 'number' ? String(value) : null; }

export function isCompatibleModel(item: OpenRouterModel, now = new Date()): boolean {
  const architecture = item.architecture as Record<string, unknown> | undefined;
  const modalities = architecture?.modality;
  const [input = '', output = ''] = typeof modalities === 'string' ? modalities.split('->') : [];
  const inputModalities = asArray(architecture?.input_modalities ?? input.split('+'));
  const outputModalities = asArray(architecture?.output_modalities ?? output.split('+'));
  const expiresAt = text(item.expires_at);
  return Boolean(
    text(item.id) && text(item.name) && Number(item.context_length) > 0 &&
    inputModalities.includes('text') && outputModalities.includes('text') &&
    asArray(item.supported_parameters).includes('structured_outputs') &&
    numeric((item.pricing as Record<string, unknown> | undefined)?.prompt) !== null &&
    numeric((item.pricing as Record<string, unknown> | undefined)?.completion) !== null &&
    (!expiresAt || new Date(expiresAt) > now),
  );
}

export class OpenRouterCatalogService {
  constructor(private readonly dataSource: PostgresDataSource) {}

  async sync(): Promise<SyncResult> {
    const key = process.env.OPENROUTER_API_KEY;
    if (!key) throw new Error('OpenRouter API key is not configured.');
    const lock = await this.dataSource.execute(`SELECT pg_try_advisory_lock(7412109) AS locked`) as Array<{locked: boolean}>;
    if (!lock[0]?.locked) throw new Error('OpenRouter catalog sync is already running.');
    const started = await this.dataSource.execute(
      `INSERT INTO llm_model_sync_runs (provider, status) VALUES ($1, 'running') RETURNING id`, [PROVIDER],
    ) as Array<{id: number}>;
    try {
      const response = await fetch(`${process.env.OPENROUTER_BASE_URL ?? 'https://openrouter.ai/api/v1'}/models/user`, {
        headers: {Authorization: `Bearer ${key}`},
      });
      if (!response.ok) throw new Error(`OpenRouter catalog request failed (${response.status}).`);
      const body = await response.json() as {data?: OpenRouterModel[]};
      const source = Array.isArray(body.data) ? body.data : [];
      const compatible = source.filter(item => isCompatibleModel(item));
      if (!compatible.length) throw new Error('OpenRouter returned no compatible models.');
      await this.dataSource.execute(`UPDATE llm_models SET is_active = FALSE, updated_at = now() WHERE provider = $1`, [PROVIDER]);
      for (const item of compatible) await this.upsert(item);
      const completedAt = new Date().toISOString();
      await this.dataSource.execute(
        `UPDATE llm_model_sync_runs SET status = 'completed', received_count = $1, active_count = $2, skipped_count = $3, completed_at = $4 WHERE id = $5`,
        [source.length, compatible.length, source.length - compatible.length, completedAt, started[0].id],
      );
      return {received: source.length, active: compatible.length, skipped: source.length - compatible.length, completedAt};
    } catch (error) {
      await this.dataSource.execute(
        `UPDATE llm_model_sync_runs SET status = 'failed', error_message = $1, completed_at = now() WHERE id = $2`,
        [error instanceof Error ? error.message.slice(0, 500) : 'Catalog synchronization failed.', started[0].id],
      );
      throw error;
    } finally {
      await this.dataSource.execute(`SELECT pg_advisory_unlock(7412109)`);
    }
  }

  private async upsert(item: OpenRouterModel): Promise<void> {
    const pricing = (item.pricing ?? {}) as Record<string, unknown>;
    const architecture = (item.architecture ?? {}) as Record<string, unknown>;
    const rawPricing = JSON.stringify(pricing);
    const pricingHash = createHash('sha256').update(rawPricing).digest('hex');
    const id = text(item.id)!;
    const values = [PROVIDER, id, text(item.canonical_slug), text(item.name)!, text(item.description), Number(item.context_length), Number(item.top_provider && (item.top_provider as Record<string, unknown>).max_completion_tokens) || null, JSON.stringify(asArray(architecture.input_modalities ?? String(architecture.modality ?? '').split('->')[0].split('+'))), JSON.stringify(asArray(architecture.output_modalities ?? String(architecture.modality ?? '').split('->')[1]?.split('+'))), JSON.stringify(asArray(item.supported_parameters)), JSON.stringify(architecture), JSON.stringify(item.reasoning ?? null), rawPricing, numeric(pricing.prompt), numeric(pricing.completion), numeric(pricing.request), numeric(pricing.input_cache_read), numeric(pricing.input_cache_write), pricingHash, text(item.expires_at)];
    await this.dataSource.execute(
      `INSERT INTO llm_models (provider, model_id, canonical_slug, display_name, description, context_length, max_completion_tokens, input_modalities, output_modalities, supported_parameters, architecture, reasoning_config, raw_pricing, prompt_usd_per_token, completion_usd_per_token, request_usd, cache_read_usd_per_token, cache_write_usd_per_token, pricing_hash, expires_at, is_active, last_synced_at)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10::jsonb,$11::jsonb,$12::jsonb,$13::jsonb,$14,$15,$16,$17,$18,$19,$20,TRUE,now())
       ON CONFLICT (provider, model_id) DO UPDATE SET canonical_slug=EXCLUDED.canonical_slug, display_name=EXCLUDED.display_name, description=EXCLUDED.description, context_length=EXCLUDED.context_length, max_completion_tokens=EXCLUDED.max_completion_tokens, input_modalities=EXCLUDED.input_modalities, output_modalities=EXCLUDED.output_modalities, supported_parameters=EXCLUDED.supported_parameters, architecture=EXCLUDED.architecture, reasoning_config=EXCLUDED.reasoning_config, raw_pricing=EXCLUDED.raw_pricing, prompt_usd_per_token=EXCLUDED.prompt_usd_per_token, completion_usd_per_token=EXCLUDED.completion_usd_per_token, request_usd=EXCLUDED.request_usd, cache_read_usd_per_token=EXCLUDED.cache_read_usd_per_token, cache_write_usd_per_token=EXCLUDED.cache_write_usd_per_token, pricing_hash=EXCLUDED.pricing_hash, expires_at=EXCLUDED.expires_at, is_active=TRUE, last_synced_at=now(), updated_at=now()`, values,
    );
    await this.dataSource.execute(
      `INSERT INTO llm_model_price_snapshots (provider, model_id, pricing_hash, raw_pricing, prompt_usd_per_token, completion_usd_per_token, request_usd, cache_read_usd_per_token, cache_write_usd_per_token) VALUES ($1,$2,$3,$4::jsonb,$5,$6,$7,$8,$9) ON CONFLICT (provider, model_id, pricing_hash) DO NOTHING`,
      [PROVIDER, id, pricingHash, rawPricing, numeric(pricing.prompt), numeric(pricing.completion), numeric(pricing.request), numeric(pricing.input_cache_read), numeric(pricing.input_cache_write)],
    );
  }
}
