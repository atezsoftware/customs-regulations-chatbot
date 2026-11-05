import type { FileDescriptor } from "@/app/app/interfaces";

export interface ASv3SessionFile extends FileDescriptor {
  token_count: number;
  size: number;
}

export interface ASv3SessionFileUpload {
  files: ASv3SessionFile[];
  rejected_files: { name: string; reason: string }[];
}
