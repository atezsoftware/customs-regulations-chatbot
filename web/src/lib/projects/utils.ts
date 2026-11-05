import type { FileDescriptor } from "@/app/app/interfaces";
import type { ProjectFile } from "@/lib/projects/types";

export function projectsFileToFileDescriptor(
  file: ProjectFile
): FileDescriptor {
  return {
    id: file.file_id,
    type: file.chat_file_type,
    name: file.name,
    ...(file.is_session_attachment
      ? { is_session_attachment: true }
      : { user_file_id: file.id }),
  };
}

export function projectFilesToFileDescriptors(
  files: ProjectFile[]
): FileDescriptor[] {
  return files.map(projectsFileToFileDescriptor);
}
