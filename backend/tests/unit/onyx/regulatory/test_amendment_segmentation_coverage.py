import json
from unittest.mock import MagicMock

import pytest

from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.regulatory.amendments.segmenter import segment_amendment_text
from onyx.regulatory.structured_llm import StructuredOutputValidationError

REPEAL = "MADDE 16- Aynı Tebliğin Ek-2’sinde yer alan listenin 26 ncı sırası yürürlükten kaldırılmıştır."
INSERT = "MADDE 17- Aynı Tebliğin Ek-2’sinde yer alan listeye aşağıdaki sıra eklenmiştir.\n26. 8429.11.00.00.00 Paletli olanlar"
COMMENCEMENT = "MADDE 22- Bu Tebliğ 1/1/2027 tarihinde yürürlüğe girer."


def _response(texts: list[str]) -> ModelResponse:
    return ModelResponse(
        id="segmentation-coverage",
        created="2026-09-21",
        choice=Choice(
            message=Message(
                content=json.dumps(
                    {
                        "instructions": [
                            {
                                "instruction_text": text,
                                "search_query": "Tebliğin Ek-2 listesi nedir?",
                                "recovery_query": "Tebliğ Ek-2",
                            }
                            for text in texts
                        ]
                    }
                )
            )
        ),
    )


def test_missing_amendment_is_repaired_without_requiring_commencement_proposal() -> (
    None
):
    llm = MagicMock()
    llm.invoke.side_effect = [_response([REPEAL]), _response([REPEAL, INSERT])]

    result = segment_amendment_text(llm, "\n\n".join([REPEAL, INSERT, COMMENCEMENT]))

    assert [item.instruction_text for item in result.instructions] == [REPEAL, INSERT]
    assert llm.invoke.call_count == 2
    assert "17" in str(llm.invoke.call_args.args[0][-1])


@pytest.mark.parametrize("line_break", [False, True])
def test_persistent_omission_cannot_be_checkpointed_as_success(
    line_break: bool,
) -> None:
    llm = MagicMock()
    llm.invoke.return_value = _response([REPEAL])
    with pytest.raises(StructuredOutputValidationError, match="17"):
        segment_amendment_text(
            llm,
            REPEAL
            + "\n\n"
            + (
                INSERT.replace("sıra eklenmiştir", "sıra\neklenmiştir")
                if line_break
                else INSERT
            ),
        )


def test_quoted_target_article_does_not_require_an_extra_proposal() -> None:
    text = "MADDE 3- Aynı Tebliğin 8 inci maddesi aşağıdaki şekilde değiştirilmiştir.\n“Madde 8- Ürünler denetlenir.”"
    llm = MagicMock()
    llm.invoke.return_value = _response([text])
    result = segment_amendment_text(llm, text)
    assert len(result.instructions) == 1
    assert llm.invoke.call_count == 1
