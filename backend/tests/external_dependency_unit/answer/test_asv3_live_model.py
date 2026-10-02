from sqlalchemy.orm import Session

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import LanguageProfile, ResearchModel, parse_json_object
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.runtime import _LANGUAGE_INSTRUCTION
from onyx.asv3.sandbox import build_sandbox_specs
from onyx.context.search.models import IndexFilters
from onyx.db.models import User
from onyx.llm.factory import get_default_llm
from onyx.llm.models import ReasoningEffort
from onyx.tracing.flows import LLMFlow


def test_configured_live_model_contextual_language_and_actual_calculation(
    db_session: Session,
) -> None:
    # Use the application's configured provider, including Vertex workload identity.
    assert db_session.is_active
    llm = get_default_llm(timeout=120)
    context = RunContext(timeout_seconds=180, research_reserve_seconds=30)
    model = ResearchModel(llm, context, reasoning_effort=ReasoningEffort.LOW)
    profile = LanguageProfile.model_validate(
        parse_json_object(
            model.invoke_text(
                _LANGUAGE_INSTRUCTION,
                "Türkçe yanıt ver. İhraç edilen 100 makine için alınan 500000 TL iadenin 20 makineye orantılı payını hesapla.",
                LLMFlow.ASV3_LANGUAGE,
                max_tokens=3500,
            )
        )
    )
    assert profile.language.startswith("tr")
    assert (
        "tools" in profile.notifications and "native_citation" in profile.notifications
    )
    public_text = " ".join(
        word for pair in profile.notifications.values() for word in pair
    )
    assert "read_provision" not in public_text and "query_corpus" not in public_text
    context.language = profile.language
    broker = CorpusBroker(User(), IndexFilters(access_control_list=[]))
    specs = [spec for spec in build_sandbox_specs(broker) if spec.name == "calculate"]
    harness = Harness(
        request='Sadece sayısal hesap: 500000 TL\'nin 100 makine içinden 20 makineye orantılı payını calculate ile hesapla. Mevzuat araştırma. Sonuç JSON olsun: {"tutar":"sayısal değer"}.',
        context=context,
        registry=CapabilityRegistry(specs),
        decide=model.decide,
    )
    result = harness.run()
    assert result.status == OutcomeStatus.FOUND
    assert any(
        receipt.call.name == "calculate"
        and receipt.outcome.status == OutcomeStatus.FOUND
        for receipt in result.receipts
    )
    assert "100000" in (result.answer or "").replace(".", "").replace(",", "")
