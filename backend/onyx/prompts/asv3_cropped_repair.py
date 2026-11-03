"""Instructions for evidence research before answer integration in exp cropped."""

CROPPED_REPAIR_RESEARCH = """Choose a single focused batch of read-only corpus actions for
the supplied deficiency. Investigate only this requested fact or governing source.
Do not inspect whether the answer says it correctly, write an answer, invent a source,
or repeat an already completed lookup. Prefer the named provision directly. Preserve
article suffixes, qualifiers, applicable dates, exceptions and cross-references.
An unavailable or ambiguous result does not prove that a rule does not exist.
Return at most four actions using the supplied capability schemas. If no useful
action exists, return an empty actions list and explain the limitation."""

CROPPED_REPAIR_CHECK = """Check only whether the supplied deficiency has been resolved
by the supplied canonical originals. This is an evidence check, not an answer review:
there is no final answer in this packet. Do not check answer coverage or wording.
For found, give the supported finding, all material conditions, exceptions, dates and
limitations, and select literal witnesses from the actual supplied originals. A lower
instrument quoting a governing provision cannot replace that governing original.
Return not_found if this focused attempt did not obtain the requested original;
return incomplete if available text cannot resolve the requested fact. Neither status
proves absence of law. Do not treat no confirming court decision as a defect unless
this particular check requested a necessary ruling. Never invent a citation or quote."""

CROPPED_REPAIR_INTEGRATION = """The focused evidence research is finished. Integrate
only its verified findings into the owned answer, correcting every affected summary,
detail, table and next step. Preserve every independently supported detail and citation.
For not_found, incomplete or failed, disclose that this attempt did not verify the
specific source or fact; do not assert its legal result or that no rule exists. Use the
supplied canonical source-gap sentence verbatim as a standalone uncited paragraph.
This deficiency cannot be researched again. Submit the corrected retained answer once,
using exact owned unit IDs and current coverage/source metadata. Do not call research
tools or rewrite unrelated units. The complete reconstructed answer still undergoes
all publication checks; an evidence check is not an approval of that answer.

Repair each flagged answer unit in place. Adding a cited explanation elsewhere does
not repair an unchanged uncited paragraph. A replacement may create several answer
units: every resulting paragraph or list item that invokes a governing provision
must carry its own adjacent supporting original citation, including caveats and
follow-up offers. Recheck the entire replacement, not only its first paragraph.
For a found governing-original check, the original was obtained; do not retain a
claim that it was not read. Distinguish a remaining substantive conflict from a
missing citation. Missing confirming case law alone creates no new uncertainty;
retain a caveat only for a concrete unresolved material effect of the sources or
scenario, and cite the provisions whose interaction is being qualified."""

CROPPED_CITATION_CORRECTION = """This is the last citation-only correction of an
otherwise retained answer. Every governing original in the supplied citation plan
was obtained and verified. Add its actual supporting [n] marker inside each flagged
owned unit. Select only from that unit's supplied original requirements, after reading
the originals. Match its supplied exact unit_texts to the actual retained-answer units;
use their exposed retained unit IDs for edits, not the diagnostic assertion IDs.
Preserve all existing citation markers, words, conditions, exceptions,
dates, conclusions, paragraphs, tables and unrelated units. Do not add a missing-source
notice, change legal wording, perform research or infer that evidence acquisition
approved an assertion. If a requirement cannot be supported without changing legal
wording, do not fabricate a citation. Submit one retained partial answer. Host checks
will reject changed assertions, unrelated citations or another unresolved source gap."""
