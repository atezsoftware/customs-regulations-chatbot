"""Own-opening document identity, distinct from citations inside provisions."""

import hashlib
import json
import re
import unicodedata

from pydantic import BaseModel, ConfigDict


class DocumentIdentityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_file_id: str
    source_chunk_id: str
    source_sha256: str
    fields: dict[str, str | None]


def _fold(value: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", value.casefold().replace("ı", "i"))
        if not unicodedata.combining(c)
    )


def derive_document_identity(
    *, user_file_id: str, source_chunk_id: str, text: str, root_heading: str
) -> DocumentIdentityEvidence | None:
    if len(text) > 16384 or not 8 <= len(root_heading) <= 600:
        return None
    title = re.sub(r"(?:\*\*|__|`)", "", root_heading).strip()
    kind_match = re.search(
        r"\b(yonetmelik|yonetmeligi|teblig|tebligi|kanun|kanunu|karar|karari|genelge|genelgesi)\s*$",
        _fold(title),
    )
    if kind_match is None:
        return None
    kind = next(
        k
        for k in ("yonetmelik", "teblig", "kanun", "karar", "genelge")
        if kind_match.group().startswith(k)
        or (k == "yonetmelik" and kind_match.group() == "yonetmeligi")
    )
    clean = re.sub(r"(?:\*\*|__|`|(?m:^\*)|(?m:\*$))", "", text).strip()
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    official = (
        len(lines) >= 6
        and lines[1] in {"Resmî Gazete", "Resmi Gazete"}
        and re.fullmatch(r"Sayı\s*:\s*\d+", lines[2]) is not None
        and _fold(lines[3]) == kind
        and lines[4].endswith("Bakanlığından:")
    )
    opening = "\n".join(lines[5:]) if official else clean
    match = re.match(r"\s+".join(re.escape(word) for word in title.split()), opening)
    if match is None:
        return None
    tail = opening[match.end() :]
    if tail and not tail[0].isspace() and tail[0] != "(":
        return None
    remainder = re.sub(r"^(?:\s*\([^\n)]{1,80}\))+", "", tail).lstrip()
    continuation = remainder.splitlines()[0] if remainder else ""
    if re.search(
        r"\b(?:degisiklik|yapilmasina|yonetmelik|teblig|kanun|karar)\b",
        _fold(continuation),
    ) and not re.match(
        r"^(?:Karar\s+Sayısı|Kanun\s+No|Genelge\s+No)\s*:", continuation, re.IGNORECASE
    ):
        return None
    if not official and (
        not tail.startswith("\n")
        or not re.search(r"Bakanlığından|Resm[iî]\s+Gazete|R\.G\.", tail[:512])
    ):
        return None
    number: str | None = None
    number_tail = re.sub(r"^\s*[^\n:]{1,160}Bakanlığından:[^\n]*\n", "", tail, count=1)
    if kind == "teblig":
        own_number = re.match(
            r"\s*(?:\((?:19|20)\d{2}/\d+/(?:AB|AT|AET)\)\s*)?\(([A-ZÇĞİÖŞÜ]{2,12}-(?:19|20)\d{2}/\d+)\)(?=\s|$)",
            number_tail,
        )
        if own_number is not None:
            number = own_number[1]
    if kind in {"karar", "kanun", "genelge"}:
        labeled = re.match(
            r"\s*(?:Karar\s+Sayısı|Kanun\s+No|Genelge\s+No)\s*:\s*(\d+(?:/\d+)?)\b",
            number_tail,
            re.IGNORECASE,
        )
        if labeled is not None:
            number = labeled[1]
    digest = hashlib.sha256(
        json.dumps(
            {
                "user_file_id": user_file_id,
                "source_chunk_id": source_chunk_id,
                "text": text,
                "root_heading": root_heading,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return DocumentIdentityEvidence(
        user_file_id=user_file_id,
        source_chunk_id=source_chunk_id,
        source_sha256=digest,
        fields={
            "document_title": title,
            "document_type": kind,
            **(
                {"document_number": number}
                if number is not None or kind == "yonetmelik"
                else {}
            ),
        },
    )
