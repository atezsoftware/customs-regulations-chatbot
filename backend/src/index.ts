import 'dotenv/config';
import {ApplicationConfig, BackendApplication} from './application';
import {PostgresDataSource} from './datasources';
import {OpenRouterCatalogService} from './modules/llm-catalog/services';

export * from './application';

export async function main(options: ApplicationConfig = {}) {
  const app = new BackendApplication(options);
  await app.boot();
  await app.start();

  // Catalog sync is intentionally best-effort: a first-boot failure leaves
  // the safe model endpoint unavailable (503), while a later failure retains
  // the last successful catalog. It must never prevent the backend itself
  // from starting.
  const syncCatalog = async () => {
    try {
      const dataSource = await app.get<PostgresDataSource>('datasources.postgres');
      await new OpenRouterCatalogService(dataSource).sync();
    } catch (error) {
      console.warn('OpenRouter catalog sync failed:', error instanceof Error ? error.message : 'unknown error');
    }
  };
  void syncCatalog();
  const syncMinutes = Math.max(1, Number(process.env.OPENROUTER_CATALOG_SYNC_MINUTES ?? 60));
  setInterval(() => void syncCatalog(), syncMinutes * 60_000).unref();

  const url = app.restServer.url;
  console.log(`Server is running at ${url}`);
  return app;
}

if (require.main === module) {
  const trustProxy = process.env.TRUST_PROXY ?? '1';
  const config = {
    rest: {
      port: +(process.env.PORT ?? 3000),
      host: process.env.HOST,
      gracePeriodForClose: 5000,
      expressSettings: {
        // Kubernetes ingress/load balancers set X-Forwarded-For. Express must
        // trust the known proxy hop so express-rate-limit can use the real
        // client IP instead of rejecting the forwarded header configuration.
        'trust proxy': /^\d+$/.test(trustProxy) ? Number(trustProxy) : trustProxy,
      },
      openApiSpec: {
        setServersFromRequest: true,
      },
    },
  };
  main(config).catch(err => {
    console.error('Cannot start the application.', err);
    process.exit(1);
  });
}
