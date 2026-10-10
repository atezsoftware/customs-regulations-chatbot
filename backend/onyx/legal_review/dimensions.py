"""Substantive research questions shared by the reader and independent reviewer."""

from onyx.legal_review.models import LegalDimension

DIMENSION_GUIDANCE: dict[LegalDimension, str] = {
    LegalDimension.LEGAL_BASIS: (
        "Identify the governing primary law and its applicable implementing provisions. "
        "Distinguish law, regulation, communique, circular and administrative opinion by "
        "hierarchy. A secondary document naming an article does not establish that unread "
        "article's full scope, conditions or continued legal effect."
    ),
    LegalDimension.VALIDITY: (
        "Establish the version applicable on the event date, including amendments, repeal, "
        "annulment, deferred effective dates and transitional provisions. Distinguish a missing "
        "event date from missing legal research. A recent read date or an unchanged older "
        "explanation does not establish the controlling norm's current validity."
    ),
    LegalDimension.CASE_LAW: (
        "Investigate judicial decisions or authoritative rulings capable of changing a material "
        "norm's validity, interpretation or application, including constitutional and administrative "
        "court decisions and relevant ministry opinions. Relevance follows from the requested "
        "legal effect; do not require unrelated precedent merely to fill this category."
    ),
    LegalDimension.EXCEPTIONS: (
        "Check exceptions, exemptions, special regimes and limiting clauses that could change "
        "the requested result. Preserve their full prerequisites and distinguish procedural "
        "facilitation from substantive exemption."
    ),
    LegalDimension.PENALTIES: (
        "Where the actual conduct or requested alternative raises a sanction, establish its "
        "statutory basis, triggering conditions, amount or calculation, applicable reductions, "
        "voluntary disclosure, settlement and limitation periods. Do not invent misconduct "
        "or incidental penalties to populate this category."
    ),
    LegalDimension.TAX: (
        "Trace the material customs duty, import VAT including its base, excise duty, additional "
        "levies, interest, refund or remission consequences. Distinguish liability, calculation, "
        "payment, security and refund; preserve partial allocations and interactions between taxes."
    ),
    LegalDimension.ALTERNATIVES: (
        "Consider other legally available regimes or procedures when they materially change a "
        "requested outcome, such as exchange, processing, refund or another declaration route. "
        "Establish eligibility rather than inferring availability from a procedure's name."
    ),
    LegalDimension.PROCEDURE: (
        "Identify the competent authority, application, approval and subsequent steps, who "
        "must act, and applicable deadlines or extensions. An earlier procedural permission "
        "does not establish a later release, discharge or refund."
    ),
    LegalDimension.DOCUMENTS: (
        "Identify the evidence and documents actually required for the applicable route, "
        "including retention, submission, forms, translations and conditional prerequisites. "
        "Distinguish physical presentation from later availability for inspection."
    ),
    LegalDimension.OPERATIONS: (
        "Explain supported practical steps, declaration fields, regime and document codes, "
        "security handling and completion conditions when needed to perform the requested "
        "procedure. Do not invent operational details beyond the governing originals."
    ),
    LegalDimension.FACTS: (
        "Identify which genuinely unknown user facts change the answer, and condition the "
        "corresponding conclusions on them. Do not treat facts already supplied as missing "
        "or substitute a request for user facts for research into obtainable law."
    ),
    LegalDimension.LIABILITY: (
        "Identify the responsible persons, material conflicting authorities and residual risks "
        "that affect the requested outcome. Distinguish established responsibility from an "
        "uncertain question requiring further professional review."
    ),
}
