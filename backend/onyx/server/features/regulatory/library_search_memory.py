"""Handle conversational turns and resolve follow-ups in a library request."""

import re

from onyx.server.features.regulatory.library_models import LibrarySearchRequest

_GREETING = (
    r"(?:merhaba|selam(?:lar)?|günaydın|iyi (?:günler|akşamlar)|"
    r"hello|hi|hey|hallo|guten (?:morgen|tag|abend))"
)
_WELLBEING = re.compile(
    rf"(?:{_GREETING} )?(?:nasılsın(?:ız)?|nasilsin(?:iz)?|naber|ne haber|iyi misin|"
    r"how are you(?: doing)?|how's it going|what's up|wie geht(?:'s|s| es dir| es ihnen))"
)
_GREETING_ONLY = re.compile(_GREETING)
_WELLBEING_REPLY = {
    "tr": "İyiyim, teşekkürler! Hangi mevzuatı araştırmak istersiniz?",
    "en": "I'm doing well, thanks! What legislation would you like to look up?",
    "de": "Mir geht es gut, danke! Welche Rechtsvorschrift möchten Sie suchen?",
}
_GREETING_REPLY = {
    "tr": "Merhaba! Hangi mevzuatı araştırmak istersiniz?",
    "en": "Hello! What legislation would you like to look up?",
    "de": "Hallo! Welche Rechtsvorschrift möchten Sie suchen?",
}

_RECALL_QUESTION = re.compile(
    r"\b(?:ne (?:sordum|sormuştum)|son (?:sorum|sorgum)|önceki (?:sorum|sorgum)|"
    r"what (?:did i (?:just )?ask|was my (?:last|previous) (?:question|query))|"
    r"was habe ich (?:gerade |zuvor )?gefragt|(?:meine )?letzte frage)\b"
)
_RECALL_OVERVIEW = re.compile(
    r"\b(?:ne (?:cevap|yanıt) verdin|önceki (?:cevap|yanıt|özet)|az önce ne buldun|"
    r"what (?:did you (?:answer|find)|was your (?:last|previous) answer)|"
    r"was hast du (?:geantwortet|gefunden)|vorherige antwort)\b"
)
_RECALL_LAWS = re.compile(
    r"\b(?:hangi mevzuat(?:ları)? (?:döndü|buldu|getirdi|çıktı)\w*|"
    r"which (?:laws|legislation) (?:did you (?:find|return)|were found)|"
    r"welche (?:gesetze|rechtsvorschriften) hast du gefunden)\b"
)
_FOLLOW_UP = re.compile(
    r"^(?:peki|(?:ve )?bu (?:kanun|yönetmelik|mevzuat|tebliğ|karar|belge)\w*|"
    r"(?:and )?(?:this|that) (?:act|law|regulation|document)|"
    r"(?:und )?dies(?:es|er|e|em) (?:gesetz|verordnung|dokument)|"
    r"\d+[a-z]?[.\s]+maddesi\b)"
)
_EXPLICIT_LAW = re.compile(
    r"\b\d{3,}\s+(?:sayılı|numaralı)\b|\b(?:law|act|gesetz)\s+\d{3,}\b"
)
_MISSING_CONTEXT = {
    "tr": "Bu arama ekranında henüz hatırlayabileceğim bir önceki sorgu yok.",
    "en": "There is no earlier query to recall in this search view.",
    "de": "In dieser Suchansicht gibt es noch keine frühere Anfrage.",
}
_MISSING_RESULT = {
    "tr": "Önceki arama için hatırlayabileceğim bir sonuç yok.",
    "en": "There is no result to recall from the previous search.",
    "de": "Für die vorherige Suche liegt kein Ergebnis vor.",
}
_QUESTION = {
    "tr": "Son sorduğunuz soru: “{question}”",
    "en": "Your last question was: “{question}”",
    "de": "Ihre letzte Frage war: „{question}“",
}


def answer_library_small_talk(request: LibrarySearchRequest) -> str | None:
    query = request.query.casefold().replace("i\u0307", "i").replace("’", "'")
    words = " ".join(re.sub(r"[.,;:!?…]+", " ", query).split())
    # Full-turn matching keeps mixed greetings and legal questions on retrieval.
    if _WELLBEING.fullmatch(words):
        return _WELLBEING_REPLY[request.language]
    if _GREETING_ONLY.fullmatch(words):
        return _GREETING_REPLY[request.language]
    return None


def recall_previous_search(request: LibrarySearchRequest) -> str | None:
    query = request.query.casefold().replace("i\u0307", "i")
    if _RECALL_QUESTION.search(query):
        return (
            _QUESTION[request.language].format(question=request.context[-1].question)
            if request.context
            else _MISSING_CONTEXT[request.language]
        )
    overview = _RECALL_OVERVIEW.search(query)
    if not overview and not _RECALL_LAWS.search(query):
        return None
    if not request.context:
        return _MISSING_CONTEXT[request.language]
    previous = request.context[-1]
    recalled = previous.overview if overview else ", ".join(previous.legislations)
    return recalled or _MISSING_RESULT[request.language]


def contextual_library_query(request: LibrarySearchRequest) -> str:
    """Resolve a reference when the preceding search identified a single law."""
    query = request.query.casefold().replace("i\u0307", "i")
    if not _FOLLOW_UP.match(query) or _EXPLICIT_LAW.search(query):
        return request.query
    previous = next(
        (turn.legislations for turn in reversed(request.context) if turn.legislations),
        [],
    )
    if len(previous) != 1 or previous[0].casefold() in query:
        return request.query
    contextual = f"{previous[0]} — {request.query}"
    return contextual if len(contextual) <= 2000 else request.query
