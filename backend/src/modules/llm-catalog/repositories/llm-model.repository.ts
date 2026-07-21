import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {LlmModel} from '../models/llm-model.model';

export class LlmModelRepository extends DefaultCrudRepository<LlmModel, typeof LlmModel.prototype.modelId> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) { super(LlmModel, dataSource); }

  async activeModels(): Promise<LlmModel[]> {
    return this.find({where: {provider: 'openrouter', isActive: true}, order: ['displayName ASC']});
  }

  async lastSuccessfulSync(): Promise<string | null> {
    const rows = await this.dataSource.execute(
      `SELECT completed_at FROM llm_model_sync_runs WHERE provider = 'openrouter' AND status = 'completed' ORDER BY completed_at DESC LIMIT 1`,
    ) as Array<{completed_at: string | null}>;
    return rows[0]?.completed_at ?? null;
  }
}
