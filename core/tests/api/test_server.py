from fs_explorer_api.server import _decode_html_entities
from fs_explorer_api.agent import _normalize_indexed_text


def test_decode_html_entities_preserves_turkish_text() -> None:
    value = "Transit s&uuml;resinin a&#351;&#305;lmas&#305; &ldquo;otomatik&rdquo; de&#287;ildir."

    assert _decode_html_entities(value) == 'Transit süresinin aşılması “otomatik” değildir.'


def test_normalize_indexed_text_decodes_entities_before_model_context() -> None:
    value = "G&uuml;mr&uuml;k &amp; transit: a&#351;&#305;lma"

    assert _normalize_indexed_text(value) == "Gümrük & transit: aşılma"
