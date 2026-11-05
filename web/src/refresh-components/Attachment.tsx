import Text from "@/refresh-components/texts/Text";
import { Button } from "@opal/components";
import { SvgDownload, SvgFileText, SvgMaximize2 } from "@opal/icons";
export interface AttachmentsProps {
  fileName: string;
  open?: () => void;
  downloadUrl?: string;
}

export default function Attachments({
  fileName,
  open,
  downloadUrl,
}: AttachmentsProps) {
  return (
    <div className="flex items-center border bg-background-tint-00 rounded-12 p-1 gap-1">
      <div className="p-2 bg-background-tint-01 rounded-08">
        <SvgFileText className="w-5 h-5 stroke-text-02" />
      </div>
      <div className="flex flex-col px-2">
        <Text as="p" secondaryAction>
          {fileName}
        </Text>
        <Text as="p" secondaryBody text03>
          Document
        </Text>
      </div>

      {downloadUrl ? (
        <Button
          aria-label="Download document"
          href={downloadUrl}
          icon={SvgDownload}
          prominence="tertiary"
          size="sm"
        />
      ) : (
        open && (
          <Button
            aria-label="Expand document"
            onClick={open}
            icon={SvgMaximize2}
            prominence="tertiary"
            size="sm"
          />
        )
      )}
    </div>
  );
}
