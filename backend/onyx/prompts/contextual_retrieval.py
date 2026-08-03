# NOTE: the prompt separation is partially done for efficiency; previously I tried
# to do it all in one prompt with sequential format() calls but this will cause a backend
# error when the document contains any {} as python will expect the {} to be filled by
# format() arguments

# ruff: noqa: E501, W605 start
CONTEXTUAL_RAG_PROMPT1 = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document"""

CONTEXTUAL_RAG_PROMPT2 = """<chunk>
{chunk}
</chunk>
Give a short, retrieval-focused context that situates this chunk within the overall document. Write in the chunk's language, or the document's primary language when the chunk is mostly identifiers or tabular data. Preserve exact names, acronyms, provision or section identifiers, codes, dates, defined terms, actors, and scope distinctions that identify what the chunk governs. Use only information supported by the document; do not apply the text to a hypothetical case or infer a legal effect that the document does not state. Answer only with the succinct context and nothing else.
""".rstrip()

CONTEXTUAL_RAG_TOKEN_ESTIMATE = 160

DOCUMENT_SUMMARY_PROMPT = """<document>
{document}
</document>
Give a short, retrieval-focused summary of the entire document in its primary language. Preserve exact document titles, identifiers, defined terms, dates, jurisdictions, actors, and scope distinctions when they help distinguish this source from similar sources. Use only information stated in the document and do not invent missing coverage. Answer only with the succinct summary and nothing else.
""".rstrip()

DOCUMENT_SUMMARY_TOKEN_ESTIMATE = 110
# ruff: noqa: E501, W605 end
