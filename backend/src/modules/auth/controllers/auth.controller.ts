import {inject} from '@loopback/core';
import {repository} from '@loopback/repository';
import {get, HttpErrors, post, Request, requestBody, response, RestBindings} from '@loopback/rest';
import {getCurrentUser} from '../../../common/auth';
import {isLocalEnv} from '../../../common/env';
import {RefreshTokenRepository, UserRepository} from '../repositories';
import {PasswordService} from '../services/password.service';
import {TokenService} from '../services/token.service';

const MAX_FAILED_ATTEMPTS = 5;
const LOCK_DURATION_MINUTES = 15;

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

interface RegisterBody {
  email: string;
  password: string;
  fullName?: string;
}

interface LoginBody {
  email: string;
  password: string;
}

interface RefreshBody {
  refreshToken: string;
}

interface ChangePasswordBody {
  currentPassword: string;
  newPassword: string;
}

interface AuthTokens {
  accessToken: string;
  refreshToken: string;
}

interface SafeUser {
  id: number;
  email: string;
  fullName?: string;
  role: string;
  // Environment property, not a per-user one — true only when NODE_ENV=local.
  // Lets the frontend hide the file-upload UI outside local without a
  // separate config endpoint.
  uploadsEnabled: boolean;
}

function toSafeUser(user: {
  id?: number;
  email: string;
  fullName?: string;
  role: string;
}): SafeUser {
  return {
    id: user.id!,
    email: user.email,
    fullName: user.fullName,
    role: user.role,
    uploadsEnabled: isLocalEnv(),
  };
}

export class AuthController {
  constructor(
    @repository(UserRepository) private userRepository: UserRepository,
    @repository(RefreshTokenRepository)
    private refreshTokenRepository: RefreshTokenRepository,
    @inject(RestBindings.Http.REQUEST) private request: Request,
  ) {}

  private passwordService = new PasswordService();
  private tokenService = new TokenService();

  private async issueTokens(user: {
    id?: number;
    email: string;
    role: string;
  }): Promise<AuthTokens> {
    const accessToken = this.tokenService.signAccessToken({
      sub: user.id!,
      email: user.email,
      role: user.role,
    });
    const {raw, hash, expiresAt} = this.tokenService.generateRefreshToken();
    await this.refreshTokenRepository.create({
      userId: user.id!,
      tokenHash: hash,
      expiresAt: expiresAt.toISOString(),
    });
    return {accessToken, refreshToken: raw};
  }

  @post('/auth/register')
  @response(200, {description: 'Registered user with auth tokens'})
  async register(
    @requestBody() body: RegisterBody,
  ): Promise<{user: SafeUser; tokens: AuthTokens}> {
    const email = body.email?.trim().toLowerCase();
    if (!email || !EMAIL_RE.test(email)) {
      throw new HttpErrors.BadRequest('A valid email is required.');
    }
    if (!body.password) {
      throw new HttpErrors.BadRequest('Password is required.');
    }

    const strength = this.passwordService.validateStrength(body.password, email);
    if (!strength.valid) {
      throw new HttpErrors.BadRequest(strength.reasons.join(' '));
    }

    const existing = await this.userRepository.findOne({where: {email}});
    if (existing) {
      throw new HttpErrors.Conflict('An account with this email already exists.');
    }

    const passwordHash = await this.passwordService.hash(body.password);
    const user = await this.userRepository.create({
      email,
      passwordHash,
      fullName: body.fullName,
      role: 'user',
    });

    const tokens = await this.issueTokens(user);
    return {user: toSafeUser(user), tokens};
  }

  @post('/auth/login')
  @response(200, {description: 'Authenticated user with auth tokens'})
  async login(@requestBody() body: LoginBody): Promise<{user: SafeUser; tokens: AuthTokens}> {
    const email = body.email?.trim().toLowerCase();
    const user = email ? await this.userRepository.findOne({where: {email}}) : null;

    // Always run a bcrypt compare, even on unknown emails, against a fixed dummy
    // hash so the response time doesn't leak whether the email exists.
    const hashToCompare =
      user?.passwordHash ?? '$2a$12$CwTycUXWue0Thq9StjUM0uJ8L6r6c2/uXyP9.6cQAR7VC4XLAwbze';
    const passwordMatches = await this.passwordService.compare(
      body.password ?? '',
      hashToCompare,
    );

    if (!user || !user.isActive) {
      throw new HttpErrors.Unauthorized('Invalid email or password.');
    }

    if (user.lockedUntil && new Date(user.lockedUntil) > new Date()) {
      throw new HttpErrors.Forbidden(
        `Account is temporarily locked. Try again after ${user.lockedUntil}.`,
      );
    }

    if (!passwordMatches) {
      const failedAttempts = (user.failedLoginAttempts ?? 0) + 1;
      const patch: Partial<typeof user> = {failedLoginAttempts: failedAttempts};
      if (failedAttempts >= MAX_FAILED_ATTEMPTS) {
        const lockedUntil = new Date();
        lockedUntil.setMinutes(lockedUntil.getMinutes() + LOCK_DURATION_MINUTES);
        patch.lockedUntil = lockedUntil.toISOString();
        patch.failedLoginAttempts = 0;
      }
      await this.userRepository.updateById(user.id, patch);
      throw new HttpErrors.Unauthorized('Invalid email or password.');
    }

    if (user.failedLoginAttempts || user.lockedUntil) {
      await this.userRepository.updateById(user.id, {
        failedLoginAttempts: 0,
        lockedUntil: null,
      });
    }

    const tokens = await this.issueTokens(user);
    return {user: toSafeUser(user), tokens};
  }

  @post('/auth/refresh')
  @response(200, {description: 'Rotated auth tokens'})
  async refresh(@requestBody() body: RefreshBody): Promise<AuthTokens> {
    if (!body.refreshToken) {
      throw new HttpErrors.BadRequest('refreshToken is required.');
    }
    const tokenHash = this.tokenService.hashRefreshToken(body.refreshToken);
    const stored = await this.refreshTokenRepository.findOne({where: {tokenHash}});

    if (!stored || stored.revokedAt || new Date(stored.expiresAt) < new Date()) {
      throw new HttpErrors.Unauthorized('Invalid or expired refresh token.');
    }

    const user = await this.userRepository.findById(stored.userId);
    if (!user.isActive) {
      throw new HttpErrors.Unauthorized('Account is no longer active.');
    }

    const tokens = await this.issueTokens(user);
    await this.refreshTokenRepository.updateById(stored.id, {
      revokedAt: new Date().toISOString(),
      replacedByTokenHash: this.tokenService.hashRefreshToken(tokens.refreshToken),
    });

    return tokens;
  }

  @post('/auth/logout')
  @response(204, {description: 'Refresh token revoked'})
  async logout(@requestBody() body: RefreshBody): Promise<void> {
    if (!body.refreshToken) return;
    const tokenHash = this.tokenService.hashRefreshToken(body.refreshToken);
    const stored = await this.refreshTokenRepository.findOne({where: {tokenHash}});
    if (stored && !stored.revokedAt) {
      await this.refreshTokenRepository.updateById(stored.id, {
        revokedAt: new Date().toISOString(),
      });
    }
  }

  @get('/auth/me')
  @response(200, {description: 'Current authenticated user'})
  async me(): Promise<SafeUser> {
    const current = await getCurrentUser(this.request, this.userRepository);
    const user = await this.userRepository.findById(current.id);
    return toSafeUser(user);
  }

  @post('/auth/change-password')
  @response(204, {description: 'Password changed'})
  async changePassword(@requestBody() body: ChangePasswordBody): Promise<void> {
    const current = await getCurrentUser(this.request, this.userRepository);
    const user = await this.userRepository.findById(current.id);

    const currentMatches = await this.passwordService.compare(
      body.currentPassword ?? '',
      user.passwordHash,
    );
    if (!currentMatches) {
      throw new HttpErrors.Unauthorized('Current password is incorrect.');
    }

    const strength = this.passwordService.validateStrength(body.newPassword, user.email);
    if (!strength.valid) {
      throw new HttpErrors.BadRequest(strength.reasons.join(' '));
    }

    const passwordHash = await this.passwordService.hash(body.newPassword);
    await this.userRepository.updateById(user.id, {passwordHash});

    // Changing the password invalidates every outstanding refresh token.
    const userTokens = await this.refreshTokenRepository.find({where: {userId: user.id}});
    const activeTokens = userTokens.filter(t => !t.revokedAt);
    await Promise.all(
      activeTokens.map(t =>
        this.refreshTokenRepository.updateById(t.id, {revokedAt: new Date().toISOString()}),
      ),
    );
  }
}
