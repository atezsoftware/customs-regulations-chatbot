import {HttpErrors, Request} from '@loopback/rest';
import {UserRepository} from '../../modules/auth/repositories';
import {TokenService} from '../../modules/auth/services';

export interface CurrentUser {
  id: number;
  email: string;
  role: string;
}

const DEV_USER_EMAIL = 'dev@local.test';
const tokenService = new TokenService();

/**
 * Resolves the authenticated user for the current request.
 *
 * In `NODE_ENV=development`, every endpoint that calls this is automatically
 * treated as a fixed local dev user — no login required. Every other
 * environment (test, production) requires and verifies a real bearer JWT.
 * This is the single place that distinction is made; nothing else should
 * re-implement it.
 */
export async function getCurrentUser(
  request: Request,
  userRepository: UserRepository,
): Promise<CurrentUser> {
  if (process.env.NODE_ENV === 'development') {
    return getOrCreateDevUser(userRepository);
  }

  const header = request.headers.authorization;
  if (!header?.startsWith('Bearer ')) {
    throw new HttpErrors.Unauthorized('Missing bearer token.');
  }
  try {
    const payload = tokenService.verifyAccessToken(header.slice('Bearer '.length));
    return {id: payload.sub, email: payload.email, role: payload.role};
  } catch {
    throw new HttpErrors.Unauthorized('Invalid or expired access token.');
  }
}

export function requireAdmin(user: CurrentUser): void {
  if (user.role !== 'admin') {
    throw new HttpErrors.Forbidden('Admin access is required.');
  }
}

async function getOrCreateDevUser(userRepository: UserRepository): Promise<CurrentUser> {
  let user = await userRepository.findOne({where: {email: DEV_USER_EMAIL}});
  if (!user) {
    user = await userRepository.create({
      email: DEV_USER_EMAIL,
      passwordHash: 'dev-bypass-no-login',
      fullName: 'Dev User',
      role: 'user',
    });
  }
  return {id: user.id!, email: user.email, role: user.role};
}
