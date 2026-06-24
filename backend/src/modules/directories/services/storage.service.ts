import fs from 'fs/promises';
import path from 'path';

const STORAGE_ROOT = process.env.STORAGE_ROOT ?? './storage';

export interface SaveFileInput {
  userName: string;
  directoryName: string;
  originalName: string;
  buffer: Buffer;
}

export interface MoveFileInput {
  storedPath: string;
  userName: string;
  directoryName: string;
  originalName: string;
}

export class StorageService {
  async saveFile(input: SaveFileInput): Promise<{storedPath: string; sizeBytes: number}> {
    const dir = this.directoryPath(input.userName, input.directoryName);
    await fs.mkdir(dir, {recursive: true});
    const storedName = await uniqueFileName(dir, safeFileName(input.originalName));
    const storedPath = path.join(dir, storedName);
    await fs.writeFile(storedPath, input.buffer);
    return {storedPath, sizeBytes: input.buffer.length};
  }

  async moveFileToDirectory(input: MoveFileInput): Promise<string> {
    const source = path.resolve(input.storedPath);
    const root = path.resolve(STORAGE_ROOT);
    if (!source.startsWith(root + path.sep)) {
      throw new Error(`Stored file path is outside STORAGE_ROOT: ${input.storedPath}`);
    }

    try {
      const stat = await fs.stat(source);
      if (!stat.isFile()) return input.storedPath;
    } catch {
      return input.storedPath;
    }

    const dir = this.directoryPath(input.userName, input.directoryName);
    await fs.mkdir(dir, {recursive: true});
    const targetName = await uniqueFileName(dir, safeFileName(input.originalName), source);
    const target = path.join(dir, targetName);
    if (source === target) return input.storedPath;
    await fs.rename(source, target);
    await removeEmptyParents(path.dirname(source), root);
    return target;
  }

  async deleteFile(storedPath: string): Promise<void> {
    await fs.rm(storedPath, {force: true});
  }

  async deleteFiles(storedPaths: string[]): Promise<void> {
    const root = path.resolve(STORAGE_ROOT);
    for (const storedPath of storedPaths) {
      const target = path.resolve(storedPath);
      if (!target.startsWith(root + path.sep)) continue;
      await fs.rm(target, {force: true});
      await removeEmptyParents(path.dirname(target), root);
    }
  }

  private directoryPath(userName: string, directoryName: string): string {
    return path.join(STORAGE_ROOT, safeSegment(userName), safeSegment(directoryName));
  }
}

async function uniqueFileName(
  dir: string,
  fileName: string,
  currentPath?: string,
): Promise<string> {
  const extension = path.extname(fileName);
  const base = path.basename(fileName, extension);
  let candidate = fileName;
  let index = 2;
  while (await pathExistsExcept(path.join(dir, candidate), currentPath)) {
    candidate = `${base}_${index}${extension}`;
    index += 1;
  }
  return candidate;
}

async function pathExistsExcept(filePath: string, currentPath?: string): Promise<boolean> {
  if (currentPath && path.resolve(filePath) === currentPath) return false;
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function safeFileName(value: string): string {
  const parsed = path.parse(value || 'file');
  const base = safeSegment(parsed.name);
  const extension = parsed.ext.replace(/[<>:"/\\|?*\x00-\x1F]+/g, '').slice(0, 32);
  return `${base}${extension}` || 'file';
}

function safeSegment(value: string): string {
  return (
    value
      .normalize('NFKD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/[<>:"/\\|?*\x00-\x1F]+/g, '_')
      .replace(/\s+/g, '_')
      .replace(/_+/g, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 120) || 'unnamed'
  );
}

async function removeEmptyParents(startDir: string, root: string): Promise<void> {
  let current = path.resolve(startDir);
  while (current.startsWith(root + path.sep) && current !== root) {
    try {
      await fs.rmdir(current);
    } catch {
      return;
    }
    current = path.dirname(current);
  }
}
