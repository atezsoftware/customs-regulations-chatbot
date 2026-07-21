import {inject} from '@loopback/core';
import {repository} from '@loopback/repository';
import {HttpErrors, post, Request, requestBody, response, RestBindings} from '@loopback/rest';
import {getCurrentUser} from '../../../common/auth';
import {UserRepository} from '../../auth/repositories';
import {PostgresDataSource} from '../../../datasources';
import {OpenRouterCatalogService} from '../services/openrouter-catalog.service';

export class AdminLlmModelsController {
  constructor(
    @repository(UserRepository) private users: UserRepository,
    @inject('datasources.postgres') private dataSource: PostgresDataSource,
    @inject(RestBindings.Http.REQUEST) private request: Request,
  ) {}

  @post('/admin/llm-models/sync')
  @response(202, {description: 'Requested OpenRouter model catalog refresh'})
  async sync(@requestBody() _body: object = {}) {
    const user = await getCurrentUser(this.request, this.users);
    if (user.role !== 'admin') throw new HttpErrors.Forbidden('Admin access is required.');
    const result = await new OpenRouterCatalogService(this.dataSource).sync();
    return {accepted: true, ...result};
  }
}
