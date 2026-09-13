"use client";

import { use } from "react";

import { SvgTag } from "@opal/icons";
import { SettingsLayouts } from "@opal/layouts";

import LabelingWorkspace from "@/app/admin/documents/sets/[documentSetId]/labeling/LabelingWorkspace";

export default function Page({
  params,
}: {
  params: Promise<{ documentSetId: string }>;
}) {
  const { documentSetId } = use(params);

  return (
    <SettingsLayouts.Root width="lg">
      <SettingsLayouts.Header
        icon={SvgTag}
        title="Chunk labeling"
        description="Label canonical chunks and review propagation to derived chunks."
        divider
        backButton
      />
      <SettingsLayouts.Body>
        <LabelingWorkspace documentSetId={Number(documentSetId)} />
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}
