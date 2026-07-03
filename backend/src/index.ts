import 'dotenv/config';
import {ApplicationConfig, BackendApplication} from './application';

export * from './application';

export async function main(options: ApplicationConfig = {}) {
  const app = new BackendApplication(options);
  await app.boot();
  await app.start();

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
