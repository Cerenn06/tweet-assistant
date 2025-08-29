from __future__ import annotations
from bs4 import BeautifulSoup, Tag
import re


REM_TAGS = {
    "script","style","noscript","svg","canvas","iframe","form",
    "header","footer","nav","aside","figcaption","amp-ad","ins"
}

REM_CLASS_PAT = re.compile(
    r"(advert|ads|ad-|ad_|sponsor|share|social|breadcrumb|tags|etiket|yorum|comments|newsletter|popup|overlay|"
    r"ticker|live-?blog|breaking|paylas|facebook|twitter|instagram|whatsapp)",
    re.I
)

REM_TEXT_PAT = re.compile(
    r"(This is a modal window|Beginning of dialog window|Escape will cancel|End of dialog window)",
    re.I
)

RELATED_KILL_PAT = re.compile(
    r"(ilginizi\s*çekebilir|ilgili|öneri|önerilen|related|recommended|trending|"
    r"most\s*read|most\s*viewed|popular|gündem|son\s*dakika|manşet|öne\s*çıkan|"
    r"haber(?:in|)\s*devam[ıi]|içeriğin\s*devam[ıi]\s*aşağ[ıi]da|"
    r"en\s*çok\s*okunan(lar|)|en\s*çok\s*i(?:z|)lenen(ler|)|"
    r"yorumlar(\s*ve\s*emojiler)?\s*aşağ[ıi]da|keşfet|sana\s*özel\s*seçtiklerimiz|son\s*eklenenler)",
    re.I
)

def _link_density(n: Tag) -> float:
    txt = (n.get_text(" ", strip=True) or "")
    if not txt:
        return 0.0
    link_txt = " ".join(a.get_text(" ", strip=True) for a in n.find_all("a"))
    return len(link_txt) / max(1, len(txt))

def clean_html(html: str) -> str:
    s = BeautifulSoup(html or "", "lxml")

    for t in list(REM_TAGS):
        for n in s.find_all(t):
            try:
                n.decompose()
            except Exception:
                pass

    nodes = list(s.find_all(True))
    for n in nodes:
        try:
            if not isinstance(n, Tag):
                continue
            role = (n.get("role") or "").lower()
            if role in {"dialog", "alert", "alertdialog"}:
                n.decompose(); continue
            if str(n.get("aria-modal", "")).lower() == "true":
                n.decompose(); continue

            classes = n.get("class")
            if isinstance(classes, (list, tuple)):
                cls = " ".join(c for c in classes if isinstance(c, str)).lower()
            elif isinstance(classes, str):
                cls = classes.lower()
            else:
                cls = ""
            idv = (n.get("id") or "").lower()

            if REM_CLASS_PAT.search(cls) or REM_CLASS_PAT.search(idv):
                n.decompose(); continue

            txt = n.get_text(" ", strip=True)
            if txt and REM_TEXT_PAT.search(txt):
                n.decompose(); continue

            attr_keys = " ".join(n.attrs.keys()).lower()
            if any(k in attr_keys for k in ["data-ad", "data-widget", "data-related"]):
                n.decompose(); continue

        except Exception:
            continue

    for n in list(s.find_all(True)):
        try:
            if not isinstance(n, Tag):
                continue
            txt = n.get_text(" ", strip=True) or ""
            if not txt:
                continue
            if RELATED_KILL_PAT.search(txt) and _link_density(n) >= 0.45:
                n.decompose()
        except Exception:
            continue

    cleaned = str(s)
    return cleaned
