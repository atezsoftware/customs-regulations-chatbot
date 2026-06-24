import crypto from 'crypto';
import jwt from 'jsonwebtoken';

const ACCESS_TOKEN_TTL = process.env.ACCESS_TOKEN_TTL ?? '15m';
const REFRESH_TOKEN_TTL_DAYS = Number(process.env.REFRESH_TOKEN_TTL_DAYS ?? 30);

function getJwtSecret(): string {
  const secret = process.env.JWT_SECRET;
  if (secret) return secret;
  if (process.env.NODE_ENV === 'production') {
    throw new Error('JWT_SECRET must be set in production.');
  }
  return 'dev-only-insecure-secret';
}

export interface AccessTokenPayload {
  sub: number;
  email: string;
  role: string;
}

export class TokenService {
  signAccessToken(payload: AccessTokenPayload): string {
    const options: jwt.SignOptions = {
      expiresIn: ACCESS_TOKEN_TTL as jwt.SignOptions['expiresIn'],
    };
    return jwt.sign(payload, getJwtSecret(), options);
  }

  verifyAccessToken(token: string): AccessTokenPayload {
    return jwt.verify(token, getJwtSecret()) as unknown as AccessTokenPayload;
  }

  /** Returns the raw refresh token (sent to the client) and its hash (stored in the DB). */
  generateRefreshToken(): {raw: string; hash: string; expiresAt: Date} {
    const raw = crypto.randomBytes(48).toString('hex');
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + REFRESH_TOKEN_TTL_DAYS);
    return {raw, hash: this.hashRefreshToken(raw), expiresAt};
  }

  hashRefreshToken(raw: string): string {
    return crypto.createHash('sha256').update(raw).digest('hex');
  }
}
