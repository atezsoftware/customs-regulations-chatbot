"""Wall-time budgets for source acquisition and page-wise visual preparation."""

SOURCE_ACQUISITION_SECONDS = 180
MAX_SOURCE_PREPARATION_SECONDS = 2 * 60 * 60
SOURCE_PACKAGE_LEASE_MARGIN_SECONDS = 5 * 60


def source_preparation_seconds(pdf_page_count: int) -> int:
    if pdf_page_count < 0:
        raise ValueError("PDF page count must not be negative")
    return min(
        MAX_SOURCE_PREPARATION_SECONDS,
        SOURCE_ACQUISITION_SECONDS + 60 + 90 * pdf_page_count,
    )
