import {inject} from '@loopback/core';
import {DefaultCrudRepository} from '@loopback/repository';
import {PostgresDataSource} from '../../../datasources';
import {ChatResearchStep, ChatResearchStepRelations} from '../models';

export class ChatResearchStepRepository extends DefaultCrudRepository<
  ChatResearchStep,
  typeof ChatResearchStep.prototype.id,
  ChatResearchStepRelations
> {
  constructor(@inject('datasources.postgres') dataSource: PostgresDataSource) {
    super(ChatResearchStep, dataSource);
  }
}
