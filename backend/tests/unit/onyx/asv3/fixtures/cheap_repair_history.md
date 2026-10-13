# Recorded retained-answer regression witnesses

`cheap_repair_history.json.gz` is lossless gzip of UTF-8 JSON. Its adjacent manifest records both hashes, size, coverage and source-artifact location. Compression applies only to the test fixture, never an LLM input. The extraction script is retained in the gitignored audit plan at `plans/experimental-sol61-cost-audit-20261012/repair-cost-design/extract_regression_fixtures.py`.

The fixture covers the twelve primary runs with coordinator rejection plus the separate 5088 historical acceptance fixture: 18 actual retained candidates and 822 selected literal witness rows across thirteen runs. It contains exact draft bodies, recorded replacement offsets/owned IDs, candidate hashes, prepared publication hashes where the same candidate reached a tool, original source/chunk/text hashes, full literals, locators and version uncertainty. It excludes the giant whole 608-original 5077 prompt snapshot. Selected witnesses include all citations in the draft/candidate and all eighteen sources supporting 5077's recorded useful edits, including the five missed by draft navigation.

Tests bind the exact captured unit text/ranges into the current real retained resolver, check the original captured unit-ID hash, and compare the complete reconstructed candidate's hash to the recorded candidate/publication. This detects lost conditions or other text changes in the captured patch; it does not establish that a new model call will produce that patch or certify legal correctness.

Scope boundaries:

- 4890, 4896 and 4905 are historical v34-exp predecessor runs. Their detailed graph hashes are retained; their results are not proof about the current release.
- 5068's reconstructed candidate failed before publication. It must never be counted as a delivered improvement.
- Three intermediate candidates in 5043/5007 were blocked on metadata before a tool ran. Their later next-tool body is not substituted for their own candidate hash.
- 5088 is a separate historical acceptance fixture rather than a user-question member of the primary twelve rejected runs.
- Some historical 5043/5088 tool records precede a later ledger upgrade to citable evidence. The fixture preserves their captured `citable=false` flags. Fresh-packet validation conservatively falls back for such drafts; resolver regression still checks their exact candidate. Tests do not invent an upgrade receipt.
- All literals are complete for their captured chunks. Full chunks, citation matches and before/after hashes do not prove full provision closure, current law or applicability.

No model call or network request was made to construct or run these tests.
