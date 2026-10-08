import pytest

from onyx.supersearch.witnesses import exact_display_witness


@pytest.mark.parametrize(
    ("quotation", "original", "expected"),
    [
        (
            "MADDE 142- Tamirat bedelsizdir.",
            "**MADDE 142-** Tamirat bedelsizdir.",
            "**MADDE 142-** Tamirat bedelsizdir.",
        ),
        (
            "tamir masrafları esas alınır",
            "**Tamir masrafları** esas alınır [7].",
            "**Tamir masrafları** esas alınır",
        ),
        ("A ve B", "A  ve\nB", "A  ve\nB"),
        ("veya", "ve ya", "veya"),
        ("şart", "Şart. ŞART.", "şart"),
        ("muafiyet uygulanır", "Muafiyet uygulanmaz", "muafiyet uygulanır"),
        ("12", "1.2", "12"),
        ("a+b", "a*b", "a+b"),
        ("1/2", "1-2", "1/2"),
    ],
)
def test_only_unique_presentation_equivalent_contiguous_spans_are_bound(
    quotation: str, original: str, expected: str
) -> None:
    assert exact_display_witness(quotation, original) == expected
