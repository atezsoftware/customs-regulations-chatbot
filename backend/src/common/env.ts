/**
 * `NODE_ENV=local` is the only environment that gets auth bypass and file
 * uploads — deliberately distinct from `NODE_ENV=development`, which names a
 * real deployed cluster (see the backend EKS workflow's `dev` environment)
 * and must behave like `test`/`production` (real auth, uploads disabled).
 */
export function isLocalEnv(): boolean {
  return process.env.NODE_ENV === 'local';
}

/**
 * Enables the dataset management UI and its mutating endpoints outside a
 * developer's machine. This must be opted into per deployment; production
 * auth remains in force and the controllers additionally require admin.
 */
export function isDatasetManagementEnabled(): boolean {
  if (isLocalEnv()) return true;
  return ['1', 'true', 'yes', 'on'].includes(
    (process.env.DATASET_MANAGEMENT_ENABLED ?? '').trim().toLowerCase(),
  );
}
