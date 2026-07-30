/**
 * `NODE_ENV=local` is the only environment that gets auth bypass and file
 * uploads — deliberately distinct from `NODE_ENV=development`, which names a
 * real deployed cluster (see the backend EKS workflow's `dev` environment)
 * and must behave like `test`/`production` (real auth, uploads disabled).
 */
export function isLocalEnv(): boolean {
  return process.env.NODE_ENV === 'local';
}
