import {BootMixin} from '@loopback/boot';
import {ApplicationConfig} from '@loopback/core';
import {RepositoryMixin} from '@loopback/repository';
import {RestApplication} from '@loopback/rest';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';

export {ApplicationConfig};

export class BackendApplication extends BootMixin(RepositoryMixin(RestApplication)) {
  constructor(options: ApplicationConfig = {}) {
    super(options);

    this.projectRoot = __dirname;
    this.bootOptions = {
      // Domain modules (auth, directories, chat, ...) each own their own
      // controllers/repositories under modules/<name>/ — booted recursively
      // so adding a module never requires touching this file again.
      controllers: {
        dirs: ['modules'],
        extensions: ['.controller.js', '.controller.ts'],
        nested: true,
      },
      datasources: {
        dirs: ['datasources'],
        extensions: ['.datasource.js', '.datasource.ts'],
        nested: true,
      },
      repositories: {
        dirs: ['modules'],
        extensions: ['.repository.js', '.repository.ts'],
        nested: true,
      },
    };

    this.expressMiddleware(helmet);
    this.expressMiddleware(rateLimit, {
      windowMs: 15 * 60 * 1000,
      max: 300,
      standardHeaders: true,
      legacyHeaders: false,
    });
  }
}
