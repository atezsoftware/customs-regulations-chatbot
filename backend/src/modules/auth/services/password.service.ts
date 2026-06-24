import bcrypt from 'bcryptjs';

const BCRYPT_ROUNDS = 12;
const MIN_LENGTH = 10;

// Small denylist of the most commonly breached/guessed passwords. Not exhaustive —
// just enough to block the obvious cases on top of the structural rules below.
const COMMON_PASSWORDS = new Set([
  'password',
  'password1',
  '123456789',
  '12345678',
  'qwerty123',
  'letmein123',
  'admin1234',
]);

export interface PasswordStrengthResult {
  valid: boolean;
  reasons: string[];
}

export class PasswordService {
  async hash(plainPassword: string): Promise<string> {
    return bcrypt.hash(plainPassword, BCRYPT_ROUNDS);
  }

  async compare(plainPassword: string, passwordHash: string): Promise<boolean> {
    return bcrypt.compare(plainPassword, passwordHash);
  }

  validateStrength(password: string, email?: string): PasswordStrengthResult {
    const reasons: string[] = [];

    if (password.length < MIN_LENGTH) {
      reasons.push(`Password must be at least ${MIN_LENGTH} characters long.`);
    }
    if (!/[a-z]/.test(password)) {
      reasons.push('Password must contain at least one lowercase letter.');
    }
    if (!/[A-Z]/.test(password)) {
      reasons.push('Password must contain at least one uppercase letter.');
    }
    if (!/\d/.test(password)) {
      reasons.push('Password must contain at least one digit.');
    }
    if (!/[^a-zA-Z0-9]/.test(password)) {
      reasons.push('Password must contain at least one special character.');
    }
    if (COMMON_PASSWORDS.has(password.toLowerCase())) {
      reasons.push('Password is too common, choose a different one.');
    }
    const emailLocalPart = email?.split('@')[0]?.toLowerCase();
    if (emailLocalPart && password.toLowerCase().includes(emailLocalPart)) {
      reasons.push('Password must not contain your email address.');
    }

    return {valid: reasons.length === 0, reasons};
  }
}
