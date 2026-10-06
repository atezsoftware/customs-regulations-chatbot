"""Owned literal fact references decode to the existing outcome validation format."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Sequence

from pydantic import JsonValue

from onyx.asv3.models import RunContext
from onyx.asv3.outcome_map import OutcomeMap
from onyx.asv3.parallel_execution import parallel_execution_enabled
from onyx.asv3.retained_answer import RETAINED_TERMINALS
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT

_TERMINALS = {"submit_answer", "submit_partial_answer", *RETAINED_TERMINALS}


def terminal_fact_references_enabled(context: RunContext) -> bool:
    return parallel_execution_enabled(context) or (
        context.services.get("asv3_workflow_variant") == ASV3_TUNED_VARIANT
        and context.services.get("research_profile") == "normal"
    )


def terminal_fact_catalogue(context: RunContext) -> list[dict[str, JsonValue]]:
    outcomes = context.services.get("outcome_map")
    if not terminal_fact_references_enabled(context) or not isinstance(
        outcomes, OutcomeMap
    ):
        return []
    scope_hash = hashlib.sha256(
        json.dumps(context.scope, sort_keys=True).encode()
    ).hexdigest()
    if outcomes.run_id != context.run_id or outcomes.scope_hash != scope_hash:
        return []
    raw = outcomes.factual_context()
    binding = json.dumps(
        [context.run_id, context.services.get("task_id"), context.scope, raw],
        ensure_ascii=False,
        sort_keys=True,
    )
    facts: list[dict[str, JsonValue]] = []
    # Every offered sentence is an exact contiguous piece of the validated context.
    for match in re.finditer(r"[^\r\n.!?]+(?:[.!?]+|(?=[\r\n]|$))", raw):
        text = match.group().strip()
        if not text:
            continue
        identifier = hashlib.sha256(
            (binding + "\0" + str(match.start()) + "\0" + text).encode()
        ).hexdigest()
        facts.append({"fact_id": "fact_" + identifier, "text": text})
    return facts


def bind_terminal_fact_references(
    tools: Sequence[dict[str, JsonValue]],
    context: RunContext,
) -> tuple[list[dict[str, JsonValue]], list[dict[str, JsonValue]]]:
    catalogue = terminal_fact_catalogue(context)
    if not catalogue:
        return list(tools), []
    definitions = copy.deepcopy(list(tools))
    bound = False
    for definition in definitions:
        function = definition.get("function")
        if not isinstance(function, dict) or function.get("name") not in _TERMINALS:
            continue
        parameters = function.get("parameters")
        properties = (
            parameters.get("properties") if isinstance(parameters, dict) else None
        )
        schema = properties.get("_outcomes") if isinstance(properties, dict) else None
        item = schema.get("items") if isinstance(schema, dict) else None
        fields = item.get("properties") if isinstance(item, dict) else None
        if not isinstance(fields, dict) or "decisive_facts" not in fields:
            continue
        fields["decisive_fact_refs"] = {
            "type": "array",
            "items": {"type": "string", "enum": [row["fact_id"] for row in catalogue]},
            "description": "Select exact supplied facts by their owned fact_id. Omit decisive_facts when using references; literal quotations remain available.",
        }
        if context.services.get("asv3_workflow_variant") == ASV3_TUNED_VARIANT:
            fields["decisive_fact_refs"]["description"] = (
                "Prefer selecting relevant supplied facts by their owned fact_id; "
                "the host expands them to exact literal scenario text. Omit "
                "decisive_facts when using references. These facts are not legal evidence."
            )
        bound = True
    return definitions, catalogue if bound else []


def normalize_terminal_fact_references(
    name: str,
    arguments: dict[str, JsonValue],
    context: RunContext,
) -> dict[str, JsonValue]:
    if name not in _TERMINALS or not terminal_fact_references_enabled(context):
        return arguments
    rows = arguments.get("_outcomes")
    if not isinstance(rows, list) or not any(
        isinstance(row, dict) and "decisive_fact_refs" in row for row in rows
    ):
        return arguments
    known = {row["fact_id"]: row["text"] for row in terminal_fact_catalogue(context)}
    normalized: list[JsonValue] = []
    for row in rows:
        if not isinstance(row, dict) or "decisive_fact_refs" not in row:
            normalized.append(row)
            continue
        refs = row["decisive_fact_refs"]
        if (
            "decisive_facts" in row
            or not isinstance(refs, list)
            or any(not isinstance(ref, str) or ref not in known for ref in refs)
        ):
            return arguments
        normalized.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key != "decisive_fact_refs"
                },
                "decisive_facts": [known[ref] for ref in refs if isinstance(ref, str)],
            }
        )
    return {**arguments, "_outcomes": normalized}
