"use client";

import { useSettings } from "@/lib/settings/hooks";
import {
  DEFAULT_APPLICATION_NAME,
  DEFAULT_LOGO_SIZE_PX,
  NEXT_PUBLIC_DO_NOT_USE_TOGGLE_OFF_DANSWER_POWERED,
} from "@/lib/constants";
import { cn } from "@opal/utils";
import Text from "@/refresh-components/texts/Text";
import Truncated from "@/refresh-components/texts/Truncated";
import { SvgOnyxLogo } from "@opal/logos";

// Ratio between logo height and the gap to the wordmark, taken from Figma.
const HEIGHT_TO_GAP_RATIO = 5 / 16;

interface LogoTypedProps {
  size: number;
  name: string;
  className?: string;
}

/** Logo mark followed by the application name as the wordmark. */
function LogoTyped({ size, name, className }: LogoTypedProps) {
  return (
    <div
      className={cn("flex flex-row items-center min-w-0", className)}
      style={{ gap: size * HEIGHT_TO_GAP_RATIO }}
    >
      <SvgOnyxLogo size={size} className="shrink-0" />
      <Truncated headingH3>{name}</Truncated>
    </div>
  );
}

export interface LogoProps {
  folded?: boolean;
  size?: number;
  className?: string;
  // Always render the product logo, ignoring enterprise white-label settings
  // (custom logo / application name). Used by product-branded surfaces like Craft.
  onyxBranded?: boolean;
}

export function Logo({ folded, size, className, onyxBranded }: LogoProps) {
  const resolvedSize = size ?? DEFAULT_LOGO_SIZE_PX;
  const { enterprise, logoUrl, appName } = useSettings();
  const logoDisplayStyle = enterprise?.logo_display_style;
  const applicationName = enterprise?.application_name;

  if (onyxBranded) {
    return folded ? (
      <SvgOnyxLogo size={resolvedSize} className={cn("shrink-0", className)} />
    ) : (
      <LogoTyped size={resolvedSize} name={appName} className={className} />
    );
  }

  const logo = logoUrl ? (
    <div
      className={cn(
        "aspect-square rounded-full overflow-hidden relative shrink-0",
        className
      )}
      style={{ height: resolvedSize }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        alt="Logo"
        src={logoUrl}
        className="object-cover object-center w-full h-full"
      />
    </div>
  ) : (
    <SvgOnyxLogo size={resolvedSize} className={cn("shrink-0", className)} />
  );

  const renderNameAndPoweredBy = (opts: {
    includeLogo: boolean;
    includeName: boolean;
  }) => {
    return (
      <div className="flex min-w-0 gap-2">
        {opts.includeLogo && logo}
        {!folded && (
          /* H3 text is 4px larger (28px) than the Logo icon (24px), so negative margin hack. */
          <div className="flex flex-1 flex-col -mt-0.5">
            {opts.includeName && (
              <Truncated headingH3>{applicationName}</Truncated>
            )}
            {!NEXT_PUBLIC_DO_NOT_USE_TOGGLE_OFF_DANSWER_POWERED &&
              !enterprise?.hide_onyx_branding && (
                <Text
                  secondaryBody
                  text03
                  className={"line-clamp-1 truncate"}
                  nowrap
                >
                  Powered by {DEFAULT_APPLICATION_NAME}
                </Text>
              )}
          </div>
        )}
      </div>
    );
  };

  // Handle "logo_only" display style
  if (logoDisplayStyle === "logo_only") {
    return renderNameAndPoweredBy({ includeLogo: true, includeName: false });
  }

  // Handle "name_only" display style
  if (logoDisplayStyle === "name_only") {
    return renderNameAndPoweredBy({ includeLogo: false, includeName: true });
  }

  // Handle "logo_and_name" or default behavior
  return applicationName ? (
    renderNameAndPoweredBy({ includeLogo: true, includeName: true })
  ) : folded ? (
    <SvgOnyxLogo size={resolvedSize} className={cn("shrink-0", className)} />
  ) : (
    <LogoTyped size={resolvedSize} name={appName} className={className} />
  );
}
