"use client";

import useSWR from "swr";
import { Text } from "@opal/components";
import type { AmendmentSourcePackage } from "@/lib/regulatory/amendments";
import { getDocumentIntakeSourceLinks } from "@/lib/regulatory/documentIntake";

const descriptions: Record<string, string> = {
  missing_base_url: "Add the original document URL to resolve this attachment.",
  "404": "The attachment was not found at its source address.",
  download_failed: "The attachment could not be downloaded.",
  depth_limit: "The attachment exceeds the supported link depth.",
  unsafe_url: "The attachment address cannot be fetched safely.",
  missing_local_zip_target:
    "An attachment referenced in the source is missing from the ZIP. Add it at the exact linked path.",
  unsafe_local_link:
    "An attachment link leaves the permitted ZIP paths. Correct the local link.",
  invalid_zip_member:
    "A ZIP member could not be read as a supported document. Check the file shown below.",
};

export default function IntakeSourceIssues({
  source,
}: {
  source: AmendmentSourcePackage;
}) {
  const { data: links, error } = useSWR(
    source.issues.length && source.manifest_sha256
      ? ["intake-source-links", source.id, source.manifest_sha256]
      : null,
    () => getDocumentIntakeSourceLinks(source.id)
  );
  return (
    <>
      {source.issues.map((issue, index) => (
        <div key={`${issue.locator}:${index}`} className="flex flex-col gap-1">
          <Text color="status-error-05">
            {descriptions[issue.code] ||
              issue.failure_detail ||
              issue.code.replaceAll("_", " ")}
          </Text>
          <Text font="secondary-body" color="text-03">
            {[issue.code.replaceAll("_", " "), issue.locator]
              .filter(Boolean)
              .join(" · ")}
          </Text>
        </div>
      ))}
      {links
        ?.filter((link) => !link.target_asset_hash)
        .map((link, index) => (
          <Text key={`${link.source_field}:${index}`} font="secondary-body">
            {[link.label, link.requested_url || link.original_url]
              .filter(Boolean)
              .join(" · ")}
          </Text>
        ))}
      {error && (
        <Text color="status-error-05">
          Attachment details could not be loaded. The source is still
          incomplete.
        </Text>
      )}
    </>
  );
}
