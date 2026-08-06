import { redirect } from "next/navigation";

export default function FilesRedirectPage() {
  redirect("/admin/documents/sets");
}
