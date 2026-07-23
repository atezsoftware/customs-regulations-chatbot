import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {BenchmarkRun, BenchmarkRunRelations} from '../models';

export class BenchmarkRunRepository extends DefaultCrudRepository<
  BenchmarkRun,
  typeof BenchmarkRun.prototype.id,
  BenchmarkRunRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(BenchmarkRun, dataSource);
  }

  /**
   * Atomically bumps completed/failed counters and flips status to
   * `completed` once every item is terminal. Done as one SQL statement
   * (not read-then-write) because multiple items from the same run can
   * finish concurrently within one orchestrator tick.
   */
  async recordItemOutcome(runId: number, outcome: 'completed' | 'failed'): Promise<void> {
    const completedDelta = outcome === 'completed' ? 1 : 0;
    const failedDelta = outcome === 'failed' ? 1 : 0;
    await this.dataSource.execute(
      `
        UPDATE benchmark_runs
        SET
          completed_items = completed_items + $2,
          failed_items = failed_items + $3,
          status = CASE
            WHEN completed_items + failed_items + $2 + $3 >= total_items THEN 'completed'
            ELSE status
          END,
          completed_at = CASE
            WHEN completed_items + failed_items + $2 + $3 >= total_items THEN now()
            ELSE completed_at
          END
        WHERE id = $1
      `,
      [runId, completedDelta, failedDelta],
    );
  }
}
