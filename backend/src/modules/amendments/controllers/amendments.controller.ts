import {inject} from '@loopback/core';
import {repository} from '@loopback/repository';
import {
  del,
  get,
  HttpErrors,
  param,
  post,
  Request,
  requestBody,
  response,
  RestBindings,
} from '@loopback/rest';
import {getCurrentUser, requireAdmin} from '../../../common/auth';
import {UserRepository} from '../../auth/repositories';
import {DirectoryRepository} from '../../directories/repositories';
import {
  analyzeAmendment,
  approveAmendmentProposals,
  CoreApiError,
  deleteAmendmentProposal,
  getAmendmentBatch,
  listAmendmentProposals,
  rejectAmendmentProposal,
} from '../services';

interface AnalyzeAmendmentBody {
  directoryId: number;
  rawText: string;
}

interface ApproveAmendmentBody {
  proposalIds: string[];
}

/**
 * Translate a `CoreApiError` (which carries core-api's real HTTP status)
 * into the matching LoopBack `HttpErrors.*`, so e.g. a 404 "no index found"
 * reaches the admin as a 404 instead of collapsing into an opaque 500 —
 * everything else falls through as a generic Error for LoopBack's default
 * (unlogged-detail) 500 handling.
 */
function rethrowCoreError(error: unknown): never {
  if (error instanceof CoreApiError) {
    if (error.status === 404) throw new HttpErrors.NotFound(error.message);
    if (error.status === 409) throw new HttpErrors.Conflict(error.message);
    if (error.status === 400) throw new HttpErrors.BadRequest(error.message);
    if (error.status === 503) throw new HttpErrors.ServiceUnavailable(error.message);
    throw new HttpErrors.BadGateway(error.message);
  }
  throw error;
}

export class AmendmentsController {
  constructor(
    @repository(UserRepository) private userRepository: UserRepository,
    @repository(DirectoryRepository) private directoryRepository: DirectoryRepository,
    @inject(RestBindings.Http.REQUEST) private request: Request,
  ) {}

  private async requireAdminUser() {
    const user = await getCurrentUser(this.request, this.userRepository);
    requireAdmin(user);
    return user;
  }

  private async requireDirectory(directoryId: number) {
    const directory = await this.directoryRepository.findOne({where: {id: directoryId}});
    if (!directory) {
      throw new HttpErrors.NotFound('Directory not found.');
    }
    return directory;
  }

  @post('/admin/amendments/analyze')
  @response(200, {description: 'Analyze pasted amendment text into reviewable proposals'})
  async analyze(@requestBody() body: AnalyzeAmendmentBody) {
    await this.requireAdminUser();
    const rawText = body.rawText?.trim();
    if (!rawText) throw new HttpErrors.BadRequest('rawText is required.');
    if (typeof body.directoryId !== 'number') {
      throw new HttpErrors.BadRequest('directoryId is required.');
    }
    await this.requireDirectory(body.directoryId);

    try {
      return await analyzeAmendment(body.directoryId, rawText);
    } catch (error) {
      rethrowCoreError(error);
    }
  }

  @get('/admin/amendments/proposals')
  @response(200, {description: 'List amendment proposals'})
  async list(
    @param.query.string('status') status?: string,
    @param.query.number('directoryId') directoryId?: number,
    @param.query.string('batchId') batchId?: string,
  ) {
    await this.requireAdminUser();
    if (directoryId !== undefined) {
      await this.requireDirectory(directoryId);
    }
    try {
      const proposals = await listAmendmentProposals({status, directoryId, batchId});
      return {proposals};
    } catch (error) {
      rethrowCoreError(error);
    }
  }

  @get('/admin/amendments/batches/{id}')
  @response(200, {description: 'Fetch one amendment batch and its proposals'})
  async getBatch(@param.path.string('id') batchId: string) {
    await this.requireAdminUser();
    try {
      return await getAmendmentBatch(batchId);
    } catch (error) {
      rethrowCoreError(error);
    }
  }

  @post('/admin/amendments/approve')
  @response(200, {description: 'Approve selected amendment proposals'})
  async approve(@requestBody() body: ApproveAmendmentBody) {
    const user = await this.requireAdminUser();
    if (!Array.isArray(body.proposalIds) || !body.proposalIds.length) {
      throw new HttpErrors.BadRequest('proposalIds is required.');
    }
    try {
      return await approveAmendmentProposals(body.proposalIds, user.email);
    } catch (error) {
      rethrowCoreError(error);
    }
  }

  @post('/admin/amendments/proposals/{id}/reject')
  @response(200, {description: 'Reject one amendment proposal'})
  async reject(@param.path.string('id') proposalId: string) {
    const user = await this.requireAdminUser();
    try {
      const proposal = await rejectAmendmentProposal(proposalId, user.email);
      return {proposal};
    } catch (error) {
      rethrowCoreError(error);
    }
  }

  @del('/admin/amendments/proposals/{id}')
  @response(200, {description: 'Delete one amendment proposal'})
  async remove(@param.path.string('id') proposalId: string) {
    await this.requireAdminUser();
    try {
      await deleteAmendmentProposal(proposalId);
      return {deleted: true};
    } catch (error) {
      rethrowCoreError(error);
    }
  }
}
