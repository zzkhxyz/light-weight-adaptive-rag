import re

from .models import Route


def language_of(text: str) -> str:
    return "ru" if len(re.findall(r"[а-яё]", text.lower())) > len(re.findall(r"[a-z]", text.lower())) else "en"


def route_query(query: str) -> Route:
    """Cheap deterministic router. Explicit route override is supported by the API."""
    q = query.strip().lower()
    if re.fullmatch(r"(hi|hello|hey|привет|здравствуй|здравствуйте|спасибо|thanks)[!?.\s]*", q):
        return Route.DIRECT
    if re.search(r"\b(what is|who is|define|explain generally|что такое|кто такой)\b", q) and not re.search(r"\b(in the document|according to|в документе|согласно|в базе)\b", q):
        return Route.DIRECT
    if re.search(r"\b(compare|synthesize|tradeoffs|why|how do|сравни|сопоставь|почему|как связаны)\b", q):
        return Route.HYDE
    if re.search(r"\b(related to|anything about|overview of|все о|что-нибудь про|обзор по)\b", q):
        return Route.MULTI_QUERY
    return Route.STANDARD_RAG


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def asks_for_overview(query: str) -> bool:
    q = query.strip().lower()
    return bool(re.search(
        r"о ч[её]м (?:этот |данный )?документ|"
        r"(?:кратко )?(?:перескажи|суммируй) (?:этот )?документ|"
        r"what is (?:this|the) document about|summari[sz]e (?:this|the) document",
        q,
    ))
