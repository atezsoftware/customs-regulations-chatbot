import {inject} from '@loopback/core';
import {repository} from '@loopback/repository';
import {
  del,
  get,
  HttpErrors,
  param,
  patch,
  post,
  Request,
  requestBody,
  response,
  RestBindings,
} from '@loopback/rest';
import {getCurrentUser, requireAdmin} from '../../../common/auth';
import {UserRepository} from '../../auth/repositories';
import {BenchmarkQuestion} from '../models';
import {
  BenchmarkQuestionDirectoryRepository,
  BenchmarkQuestionRepository,
  BenchmarkRunItemRepository,
} from '../repositories';

interface QuestionBody {
  prompt?: string;
  referenceAnswer?: string | null;
  expectedFacts?: string[] | null;
  rubricNotes?: string | null;
  tags?: string[] | null;
  isActive?: boolean;
  directoryIds?: number[];
}

function toSafeQuestion(question: BenchmarkQuestion, directoryIds: number[]) {
  return {
    id: question.id,
    prompt: question.prompt,
    referenceAnswer: question.referenceAnswer ?? null,
    expectedFacts: question.expectedFacts ?? [],
    rubricNotes: question.rubricNotes ?? null,
    tags: question.tags ?? [],
    isActive: question.isActive,
    directoryIds,
    createdAt: question.createdAt,
    updatedAt: question.updatedAt,
  };
}

export class BenchmarkQuestionsController {
  constructor(
    @repository(BenchmarkQuestionRepository) private questions: BenchmarkQuestionRepository,
    @repository(BenchmarkQuestionDirectoryRepository)
    private questionDirectories: BenchmarkQuestionDirectoryRepository,
    @repository(BenchmarkRunItemRepository) private runItems: BenchmarkRunItemRepository,
    @repository(UserRepository) private userRepository: UserRepository,
    @inject(RestBindings.Http.REQUEST) private request: Request,
  ) {}

  @get('/admin/benchmark/questions')
  @response(200, {description: 'List all benchmark questions'})
  async list() {
    const user = await getCurrentUser(this.request, this.userRepository);
    requireAdmin(user);

    const [questions, links] = await Promise.all([
      this.questions.find({order: ['id DESC']}),
      this.questionDirectories.find(),
    ]);
    const directoryIdsByQuestion = new Map<number, number[]>();
    for (const link of links) {
      const list = directoryIdsByQuestion.get(link.questionId) ?? [];
      list.push(link.directoryId);
      directoryIdsByQuestion.set(link.questionId, list);
    }

    return {
      questions: questions.map(question =>
        toSafeQuestion(question, directoryIdsByQuestion.get(question.id!) ?? []),
      ),
    };
  }

  @post('/admin/benchmark/questions')
  @response(200, {description: 'Created benchmark question'})
  async create(@requestBody() body: QuestionBody) {
    const user = await getCurrentUser(this.request, this.userRepository);
    requireAdmin(user);

    const prompt = body.prompt?.trim();
    if (!prompt) throw new HttpErrors.BadRequest('prompt is required.');

    const question = await this.questions.create({
      prompt,
      referenceAnswer: body.referenceAnswer?.trim() || undefined,
      expectedFacts: body.expectedFacts ?? undefined,
      rubricNotes: body.rubricNotes?.trim() || undefined,
      tags: body.tags ?? undefined,
      isActive: body.isActive ?? true,
      createdBy: user.id,
      updatedBy: user.id,
    });

    const directoryIds = await this.linkDirectories(question.id!, body.directoryIds ?? []);
    return toSafeQuestion(question, directoryIds);
  }

  @patch('/admin/benchmark/questions/{id}')
  @response(200, {description: 'Updated benchmark question'})
  async update(@param.path.number('id') id: number, @requestBody() body: QuestionBody) {
    const user = await getCurrentUser(this.request, this.userRepository);
    requireAdmin(user);

    const question = await this.questions.findById(id).catch(() => null);
    if (!question) throw new HttpErrors.NotFound('Benchmark question not found.');

    const patchData: Partial<BenchmarkQuestion> = {
      updatedBy: user.id,
      updatedAt: new Date().toISOString(),
    };
    if (body.prompt !== undefined) {
      const prompt = body.prompt.trim();
      if (!prompt) throw new HttpErrors.BadRequest('prompt cannot be empty.');
      patchData.prompt = prompt;
    }
    if (body.referenceAnswer !== undefined) patchData.referenceAnswer = body.referenceAnswer?.trim() || undefined;
    if (body.expectedFacts !== undefined) patchData.expectedFacts = body.expectedFacts ?? undefined;
    if (body.rubricNotes !== undefined) patchData.rubricNotes = body.rubricNotes?.trim() || undefined;
    if (body.tags !== undefined) patchData.tags = body.tags ?? undefined;
    if (body.isActive !== undefined) patchData.isActive = body.isActive;

    await this.questions.updateById(id, patchData);

    const directoryIds =
      body.directoryIds !== undefined
        ? await this.linkDirectories(id, body.directoryIds, {replace: true})
        : (await this.questionDirectories.find({where: {questionId: id}})).map(link => link.directoryId);

    const updated = await this.questions.findById(id);
    return toSafeQuestion(updated, directoryIds);
  }

  @del('/admin/benchmark/questions/{id}')
  @response(204, {description: 'Deleted benchmark question'})
  async delete(@param.path.number('id') id: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    requireAdmin(user);

    const question = await this.questions.findById(id).catch(() => null);
    if (!question) throw new HttpErrors.NotFound('Benchmark question not found.');

    const usedInRuns = await this.runItems.count({questionId: id});
    if (usedInRuns.count > 0) {
      throw new HttpErrors.Conflict(
        'This question has historical benchmark results and cannot be deleted. Deactivate it instead.',
      );
    }

    await this.questionDirectories.deleteAll({questionId: id});
    await this.questions.deleteById(id);
  }

  private async linkDirectories(
    questionId: number,
    directoryIds: number[],
    options: {replace?: boolean} = {},
  ): Promise<number[]> {
    if (options.replace) {
      await this.questionDirectories.deleteAll({questionId});
    }
    const uniqueIds = Array.from(new Set(directoryIds));
    for (const directoryId of uniqueIds) {
      await this.questionDirectories.create({questionId, directoryId});
    }
    return uniqueIds;
  }
}
