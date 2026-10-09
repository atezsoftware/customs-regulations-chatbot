"""Freeze contracts consumed by other workflows while Legal Composite evolves separately."""

import ast
import inspect
import json
from hashlib import sha256
from typing import Any

import pytest
from pydantic import BaseModel

from onyx.legal_composite import dependencies, engine, models

# Contracts from the shared workflow baseline 7ccbf6fb226725446c5c6060ce3a445fd0407455.
_SCHEMAS = {
    "WorkflowPolicy": "f57b30143c48227ecef213cef77db7a45f25a66ea675d61327d1f17a7268a67c",
    "StrictModel": "bf3309024ba1097196dd234c076016144cffbc6cb870bce0273ab47aaadecaef",
    "ResearchNeed": "885eb6f2637175773617c48ae19c99fdb82535ab7fae33845a364e7e9aa438ba",
    "SourceAction": "cc163423069139958e36ae2dab98fdd6a0291465e4417e07051188619d5e0ed3",
    "ResearchPlan": "403d565bcf3040c8b9c2d9173e80b507e66c07d5faec731824cdef30b66cb987",
    "ResearchStep": "717af4533445d287880b9b978cb17235aea6b7c7daa012adbec4f565a884dba1",
    "DraftAnswer": "f24cdd907b3c44f74babc95e5604a9f18fb110e01fda0e5976b9ed2bb7017bb9",
    "PassageSupport": "3a5a55f2266d8f78db225c455ab9169275e1aa7b37555bf072ad8b41c151f03f",
    "ConditionReview": "1a4db9e63fc7d9cbf1fae9f0a7830071508a2f04c2344193de13e8f129f4641d",
    "NeedReview": "a16ae8adebae20e5e0c2ef1e6f5fbacb3b93dd0cb81038bb884eb0050a568d00",
    "DependencyOrigin": "832f389f044fdd9c542f168e14640e5c412b559db9e144190bb752eec5eec0ef",
    "AuthorityDependency": "bcefcbd9139efb8f7ee98dc1cc131de7e295bb5f28f7f2730afc2d4dd0ec08f1",
    "DependencyWitness": "681c162303668e89c698a292e8dfd55967481fe87a9d520b004742e0b70dec0e",
    "DependencyAssessment": "4cc8c93c89118e0ee05433a4ee27d1d757ed6100efecf4eef412ea89194be7af",
    "AnswerReview": "a7f1c4ed121efb6fbc1c6603e709946f546647b1de398278e6b43fbab4c3f989",
    "WorkflowResult": "dda6e9b13fb523eb1c0ee5600478f7be2574cfac5ac7ee198c422dea5ea5ee9d",
}
_HELPERS = {
    "_normalized_name": "f79e2431c4d0f5683987b9c7db14bdbaba32903c85eca1e2f01639cf6e9842ad",
    "_literal_names": "273671adecb0f1ee2a36817ed436f700be98f31d52db852b1a604c25d0229a2f",
    "_entity_key": "07233ab14cd554e71a078014394ad09fd61a0f31cf827d5eebb494bac928cedd",  # pragma: allowlist secret
    "DependencyExpander": "06e63e36a772ef1b77f43c2117deaee35cef08fbedbb735570ea9f96a8ea2e19",
    "assess_dependencies": "abed2ff9d6fff21fad7fb48c96a70e3cd502e692c08a5a2b54e8c3ba22b5e544",
    "dependency_required_citations": "4bea29642e15b0eb688879b72d7ccefc17081401844e18e1e38ce71d46fb5ec2",
}


@pytest.mark.parametrize("name,expected", _SCHEMAS.items())
def test_shared_model_schema_remains_exact(name: str, expected: str) -> None:
    model: type[BaseModel] = getattr(models, name)
    schema = json.dumps(
        model.model_json_schema(), sort_keys=True, separators=(",", ":")
    )
    assert sha256(schema.encode()).hexdigest() == expected


def _source_fingerprint(value: Any) -> str:
    source = ast.parse(inspect.getsource(value)).body[0]
    return sha256(ast.dump(source, include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize("name,expected", _HELPERS.items())
def test_shared_dependency_behavior_remains_exact(name: str, expected: str) -> None:
    assert _source_fingerprint(getattr(dependencies, name)) == expected


def test_supersearch_shared_review_gate_remains_exact() -> None:
    from onyx.supersearch import engine as supersearch_engine

    assert supersearch_engine.review_assessment is engine.review_assessment
    assert (
        _source_fingerprint(engine.review_assessment)
        == "8e9739bb90b92c5268cf3fce69528c1d452bdd115b890265de51767136b11ea3"
    )
