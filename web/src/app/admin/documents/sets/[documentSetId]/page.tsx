"use client";
import { use } from "react";

import { ErrorCallout } from "@/components/ErrorCallout";
import { refreshDocumentSets, useDocumentSets } from "../hooks";
import { useConnectorStatus, useUserGroups } from "@/lib/hooks";
import { PageLoader, SettingsLayouts } from "@opal/layouts";
import { Tabs } from "@opal/components";
import { SvgFiles, SvgSettings } from "@opal/icons";
import { ADMIN_ROUTES } from "@/lib/admin-routes";
import CardSection from "@/components/admin/CardSection";
import { DocumentSetCreationForm } from "../DocumentSetCreationForm";
import DocumentSetFiles from "@/sections/document-sets/DocumentSetFiles";
import { useUser } from "@/providers/UserProvider";
import { useRouter } from "next/navigation";
import { useSettings } from "@/lib/settings/hooks";

const route = ADMIN_ROUTES.DOCUMENT_SETS;

function Main({ documentSetId }: { documentSetId: number }) {
  const router = useRouter();
  const { isAdmin } = useUser();
  const { vectorDbEnabled } = useSettings();

  const {
    data: documentSets,
    isLoading: isDocumentSetsLoading,
    error: documentSetsError,
  } = useDocumentSets(true);

  const {
    data: ccPairs,
    isLoading: isCCPairsLoading,
    error: ccPairsError,
  } = useConnectorStatus(30000, vectorDbEnabled);

  // EE only
  const { data: userGroups, isLoading: userGroupsIsLoading } = useUserGroups();

  if (
    isDocumentSetsLoading ||
    (vectorDbEnabled && isCCPairsLoading) ||
    userGroupsIsLoading
  ) {
    return (
      <div className="flex justify-center items-center min-h-[400px]">
        <PageLoader />
      </div>
    );
  }

  if (documentSetsError || !documentSets) {
    return (
      <ErrorCallout
        errorTitle="Failed to fetch document sets"
        errorMsg={documentSetsError}
      />
    );
  }

  if (vectorDbEnabled && (ccPairsError || !ccPairs)) {
    return (
      <ErrorCallout
        errorTitle="Failed to fetch Connectors"
        errorMsg={ccPairsError}
      />
    );
  }

  const documentSet = documentSets.find(
    (documentSet) => documentSet.id === documentSetId
  );
  if (!documentSet) {
    return (
      <ErrorCallout
        errorTitle="Document set not found"
        errorMsg={`Document set with id ${documentSetId} not found`}
      />
    );
  }

  return (
    <Tabs defaultValue={isAdmin ? "files" : "settings"}>
      <Tabs.List>
        {isAdmin && (
          <Tabs.Trigger value="files" icon={SvgFiles}>
            Files
          </Tabs.Trigger>
        )}
        <Tabs.Trigger value="settings" icon={SvgSettings}>
          Settings & Connectors
        </Tabs.Trigger>
      </Tabs.List>

      {isAdmin && (
        <Tabs.Content value="files">
          <CardSection className="mt-4">
            <DocumentSetFiles
              documentSetId={documentSet.id}
              documentSetName={documentSet.name}
            />
          </CardSection>
        </Tabs.Content>
      )}

      <Tabs.Content value="settings">
        <CardSection className="mt-4">
          <DocumentSetCreationForm
            ccPairs={ccPairs ?? []}
            userGroups={userGroups}
            onClose={() => {
              refreshDocumentSets();
              router.push("/admin/documents/sets");
            }}
            existingDocumentSet={documentSet}
          />
        </CardSection>
      </Tabs.Content>
    </Tabs>
  );
}

export default function Page(props: {
  params: Promise<{ documentSetId: string }>;
}) {
  const params = use(props.params);
  const documentSetId = parseInt(params.documentSetId);

  return (
    <SettingsLayouts.Root>
      <SettingsLayouts.Header
        icon={route.icon}
        title="Document Set"
        divider
        backButton
      />
      <SettingsLayouts.Body>
        <Main documentSetId={documentSetId} />
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}
