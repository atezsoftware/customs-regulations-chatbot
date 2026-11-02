"""Source-bound single-attempt semantic repair contracts."""

import json

from onyx.regulatory.labeling.intake_abstention_repair import (
    ABSTENTION_REPAIR_KEY,
    validate_abstention_repair,
)
from onyx.regulatory.labeling.provider import (
    validate_labeling_response,
)
from tests.unit.onyx.regulatory.labeling.abstention_repair_test_helpers import (
    REJECTED,
    TEXT,
    audit,
    build,
    response,
    taxonomy,
    validate_audit,
)


def test_reconsideration_keeps_entire_frozen_request_and_does_not_force_all_facets() -> (
    None
):
    repair = build()
    before = json.loads(repair.base_request.prompt)
    after = json.loads(repair.request.prompt)
    assert {
        key: value for key, value in after.items() if key != ABSTENTION_REPAIR_KEY
    } == before
    assert repair.request.generation_config == repair.base_request.generation_config
    assert after[ABSTENTION_REPAIR_KEY]["attempt"] == 1
    assert after[ABSTENTION_REPAIR_KEY]["rejected_response"] == REJECTED
    assert build() == repair
    result = validate_abstention_repair(
        response(), repair=repair, text=TEXT, taxonomy=taxonomy()
    )
    assert [item.label_id for item in result.labels] == ["origin"]
    assert result == validate_labeling_response(
        response(), text=TEXT, taxonomy=taxonomy()
    )
    assert validate_audit(audit(repair, response()), repair).outcome == result
    assert repair.request.system_instruction is not None
    assert "do not force a label" in repair.request.system_instruction
    assert "governing predicate" in repair.request.system_instruction
