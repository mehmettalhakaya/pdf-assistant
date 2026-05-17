from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from html import escape
from typing import Dict, List, Optional
from urllib.parse import quote

import fitz
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq

st.set_page_config(
    page_title="PDF Assistant — mtk.",
    page_icon="📄",
    layout="wide",
    # 'auto': desktop'ta açık, mobilde kapalı başlar — Streamlit zaten
    # ekran boyutuna göre karar verir. force_open_sidebar() desktop'ta
    # ek olarak garantili açar.
    initial_sidebar_state="auto",
)

load_dotenv()

st.set_option("client.toolbarMode", "minimal")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Sabit modeller (arayüzde gösterilmiyor)
# Sohbet için kalite öncelikli 70B kullanılır.
# Özet için iki katmanlı fallback — pratikte denenmiş, güvenilir:
#   1) llama-3.1-8b-instant    — Groq'ta ~700-1000 tok/s, kararlı birincil özet modeli
#   2) llama-3.3-70b-versatile — ilk başarısız olursa kalite garantisi olarak son çare
# (Daha küçük "preview" modeller bazı hesaplarda erişilebilir değil; onları
#  denemek pratiğe baktığımızda kazandırdığından daha çok zaman kaybettiriyordu.)
# Başarısız olan bir model session boyunca hatırlanır → ikinci tıklamada baştan
# 70B denenir, zaman kaybedilmez.
MODEL_NAME = "llama-3.3-70b-versatile"
CHAT_MODEL_NAME = MODEL_NAME
FAST_SUMMARY_MODEL: Optional[str] = None  # Devre dışı — bkz. yukarı
SUMMARY_MODEL_NAME = "llama-3.1-8b-instant"

# Rate limit'e takılmamak için parça boyutu ve istekler arası bekleme
CHUNK_CHAR_LIMIT = 12000         # daha az LLM çağrısı için büyük parçalar
REQUEST_PACING_SEC = 0.0         # gereksiz sabit bekleme yok; retry gerektiğinde bekler
MAX_RETRIES = 4                  # 413/429/5xx/connection durumunda deneme sayısı
MAX_TOTAL_RETRY_WAIT_SEC = 45.0  # tek bir LLM çağrısında toplam bekleme tavanı
MAX_SINGLE_WAIT_SEC = 20.0       # tek seferlik bekleme tavanı
LLM_REQUEST_TIMEOUT_SEC = 45.0   # HTTP/stream okumaları sonsuza kadar askıda kalmasın
SUMMARY_CHUNK_MAX_TOKENS = 900
SUMMARY_GROUP_MAX_TOKENS = 1600
SUMMARY_FINAL_MAX_TOKENS = 2500     # output ~2500 tok @ 3B-preview ~1500/s ≈ 1.7 sn (8B ~3.6 sn)
SUMMARY_CONTINUATION_MAX_TOKENS = 1200
DIRECT_SUMMARY_CHAR_LIMIT = 90000      # 8B-instant 128K context — daha çok PDF tek geçişte özetlenir
CHAT_CONTEXT_CHAR_LIMIT = 12000
MAX_PARALLEL_CHUNK_REQUESTS = 5        # parça özetlerini paralel istemekle 5x'e kadar hızlanma
SUMMARY_PIPELINE_VERSION = "summary-v5-stream"

# PDF güvenlik sınırları (kullanıcıya net hata için)
MAX_PDF_BYTES = 60 * 1024 * 1024  # 60 MB
MAX_PDF_PAGES = 800
MAX_PDF_TEXT_CHARS = 1_500_000    # ~ 500k token; üstü için uyar

DEFAULT_LANG = "tr"

# Flag SVG'leri doğrudan gömülü — dış dosyaya bağımlı değil
FLAG_SVGS = {
    "tr": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 20">'
        '<rect width="30" height="20" rx="3" fill="#E30A17"/>'
        '<circle cx="11" cy="10" r="6" fill="#fff"/>'
        '<circle cx="12.5" cy="10" r="4.8" fill="#E30A17"/>'
        '<polygon points="16,10 13.5,8.2 14.8,10.8 13.5,11.8 14.8,9.2" fill="#fff" transform="rotate(18,15,10)"/>'
        '</svg>'
    ),
    "en": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 20">'
        '<rect width="30" height="20" rx="3" fill="#012169"/>'
        '<path d="M0,0 L30,20 M30,0 L0,20" stroke="#fff" stroke-width="4"/>'
        '<path d="M0,0 L30,20 M30,0 L0,20" stroke="#C8102E" stroke-width="2"/>'
        '<path d="M15,0 V20 M0,10 H30" stroke="#fff" stroke-width="6"/>'
        '<path d="M15,0 V20 M0,10 H30" stroke="#C8102E" stroke-width="3"/>'
        '</svg>'
    ),
}


def svg_to_data_uri(svg: str) -> str:
    return f"data:image/svg+xml;utf8,{quote(svg.strip())}"


FLAG_DATA_URIS = {code: svg_to_data_uri(svg) for code, svg in FLAG_SVGS.items()}

LANGUAGE_NAMES = {
    "tr": {"tr": "Türkçe", "en": "Turkish"},
    "en": {"tr": "İngilizce", "en": "English"},
}

LANG = {
    "tr": {
        "language_label": "Dil",
        "upload": "PDF yükle",
        "clear": "Temizle",
        "pages": "Sayfa",
        "chars": "Karakter",
        "hero_title": "PDF <em>Assistant</em>",
        "hero_label": "MTK · AI · PDF",
        "summary_tab": "Özet",
        "chat_tab": "Sohbet",
        "btn_summary": "Özet oluştur",
        "btn_download_summary": "Özeti PDF indir",
        "btn_ask": "Sor",
        "summary_pdf_title": "PDF Özeti",
        "placeholder_chat": "PDF hakkında bir soru yaz...",
        "question_label": "Soru",
        "processing": "İşleniyor...",
        "no_pdf": "PDF yükle.",
        "no_question": "Bir soru yaz.",
        "no_api_key": "API anahtarı yok. .env dosyanı kontrol et.",
        "err_too_large": "PDF çok büyük. En fazla {limit} MB destekleniyor.",
        "err_too_many_pages": "PDF çok uzun. En fazla {limit} sayfa destekleniyor.",
        "err_encrypted": "PDF şifreli. Önce şifreyi kaldırıp tekrar yükle.",
        "err_empty": "PDF'ten metin çıkarılamadı. Tarama tabanlı PDF olabilir; OCR uygulanmış bir kopya yükle.",
        "err_open_failed": "PDF açılamadı: {detail}",
        "warn_huge_text": "PDF çok yoğun ({chars} karakter). Özet uzun sürebilir; kahveni hazırla.",
        "warn_low_quality_pdf": "PDF'in yazı katmanı kısmen bozuk (taranmış veya özel font). Bazı karakterler eksik veya '?' görünebilir, özet kalitesi düşebilir.",
        "llm_failed": "Yanıt alınamadı. Lütfen birkaç saniye sonra tekrar dene.",
        "llm_partial": "Bazı parçalar alınamadı, kalanlardan özet üretildi.",
        "llm_empty": "Model boş yanıt döndürdü. Soruyu yeniden ifade etmeyi dene.",
        "error_detail": "Hata detayı (geliştirici)",
        "fallback_notice": "Hızlı model yanıt vermedi, daha kararlı modelle yeniden deneniyor...",
        "timer_estimate": "Tahmini ~{secs} sn",
        "timer_remaining": "~{secs} sn kaldı",
        "timer_overrun": "+{secs} sn (model yavaş yanıt veriyor)",
        "timer_starting": "Başlatılıyor... (~{secs} sn)",
        "tts_play": "Sesli oku",
        "tts_resume": "Devam",
        "tts_pause": "Duraklat",
        "tts_stop": "Durdur",
        "tts_speaking": "Okunuyor",
        "tts_paused": "Duraklatıldı",
        "tts_error": "Sesli okuma hatası",
        "tts_speed": "Hız",
    },
    "en": {
        "language_label": "Language",
        "upload": "Upload PDF",
        "clear": "Clear",
        "pages": "Pages",
        "chars": "Chars",
        "hero_title": "PDF <em>Assistant</em>",
        "hero_label": "MTK · AI · PDF",
        "summary_tab": "Summary",
        "chat_tab": "Chat",
        "btn_summary": "Generate summary",
        "btn_download_summary": "Download summary PDF",
        "btn_ask": "Ask",
        "summary_pdf_title": "PDF Summary",
        "placeholder_chat": "Ask a question about the PDF...",
        "question_label": "Question",
        "processing": "Processing...",
        "no_pdf": "Upload a PDF.",
        "no_question": "Type a question.",
        "no_api_key": "API key missing. Check your .env file.",
        "err_too_large": "PDF too large. Maximum supported size is {limit} MB.",
        "err_too_many_pages": "PDF too long. Maximum {limit} pages supported.",
        "err_encrypted": "PDF is encrypted. Remove the password and re-upload.",
        "err_empty": "No text could be extracted. The PDF may be scanned/image-only; upload an OCR'd copy.",
        "err_open_failed": "Failed to open PDF: {detail}",
        "warn_huge_text": "PDF is dense ({chars} chars). Summary may take a while.",
        "warn_low_quality_pdf": "PDF text layer is partly broken (scanned or custom font). Some characters may be missing or appear as '?', summary quality may drop.",
        "llm_failed": "Failed to get a response. Please try again in a few seconds.",
        "llm_partial": "Some chunks failed; summary built from the rest.",
        "llm_empty": "Model returned an empty response. Try rephrasing your question.",
        "error_detail": "Error detail (developer)",
        "fallback_notice": "Fast model didn't respond, retrying with a more reliable model...",
        "timer_estimate": "Estimated ~{secs}s",
        "timer_remaining": "~{secs}s remaining",
        "timer_overrun": "+{secs}s (model is slow)",
        "timer_starting": "Starting... (~{secs}s)",
        "tts_play": "Read aloud",
        "tts_resume": "Resume",
        "tts_pause": "Pause",
        "tts_stop": "Stop",
        "tts_speaking": "Speaking",
        "tts_paused": "Paused",
        "tts_error": "Speech synthesis error",
        "tts_speed": "Speed",
    },
}


def get_lang() -> str:
    selected_lang = st.session_state.get("lang_selector")
    if selected_lang in LANG:
        st.session_state.lang = selected_lang
    elif "lang" not in st.session_state:
        st.session_state.lang = DEFAULT_LANG
    if st.session_state.get("lang_selector") not in LANG:
        st.session_state.lang_selector = st.session_state.lang
    return st.session_state.lang


def t(key: str) -> str:
    return LANG[get_lang()].get(key, key)


def ensure_state() -> None:
    defaults = {
        "lang": DEFAULT_LANG,
        "lang_selector": DEFAULT_LANG,
        "chunks": None,
        "summary": None,
        "chat_history": [],
        "summary_cache": {},
        "summary_pdf_cache": {},
        "chat_cache": {},
        "last_file_signature": None,
        "last_file_name": None,
        "failed_summary_models": set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_language_name(code: str) -> str:
    return LANGUAGE_NAMES[code][get_lang()]


def handle_language_change() -> None:
    selected_lang = st.session_state.get("lang_selector")
    if selected_lang in LANG:
        st.session_state.lang = selected_lang


def clear_workspace() -> None:
    st.session_state.chunks = None
    st.session_state.summary = None
    st.session_state.chat_history = []
    st.session_state.summary_cache = {}
    st.session_state.summary_pdf_cache = {}
    st.session_state.chat_cache = {}
    st.session_state.last_file_signature = None
    st.session_state.last_file_name = None


def current_document_key() -> str:
    signature = st.session_state.get("last_file_signature") or "no-file"
    return f"{signature}:{get_lang()}"


def summary_cache_key() -> str:
    return f"{current_document_key()}:{SUMMARY_PIPELINE_VERSION}"


def summary_pdf_cache_key(summary: str) -> str:
    summary_hash = hashlib.sha1(summary.encode("utf-8")).hexdigest()[:16]
    return f"{summary_cache_key()}:pdf:{summary_hash}"


def _normalize_question_for_cache(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def chat_cache_key(question: str, history: Optional[List[Dict]] = None) -> str:
    recent = history[-3:] if history else []
    history_fingerprint = hashlib.sha1(
        repr([(turn.get("q", ""), turn.get("a", "")) for turn in recent]).encode("utf-8")
    ).hexdigest()[:12]
    question_fingerprint = hashlib.sha1(
        _normalize_question_for_cache(question).encode("utf-8")
    ).hexdigest()[:12]
    return f"{current_document_key()}:chat:{history_fingerprint}:{question_fingerprint}"


class PDFLoadError(Exception):
    """PDF okuma sırasında oluşan, kullanıcıya gösterilecek hata."""

    def __init__(self, key: str, **fmt):
        self.key = key
        self.fmt = fmt
        super().__init__(key)


# Unicode "replacement character" — broken encoding'lerde bozuk glif olarak görünür.
# Ayrıca PDF'lerden gelen private-use area (PUA) karakterleri (-),
# Tagging/Tag/Variation Selector karakterleri "kare" olarak görünür.
_PUA_RE = re.compile(r"[-￰-￿�]")
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_GARBAGE_THRESHOLD = 0.30  # %30 üstü "garbage" → uyarı


def _garbage_ratio(text: str) -> float:
    """Metindeki bozuk/PUA/kontrol karakter oranı (0.0 - 1.0)."""
    if not text:
        return 1.0
    bad = len(_PUA_RE.findall(text)) + len(_CTRL_RE.findall(text)) + text.count("?")
    return min(1.0, bad / max(1, len(text)))


def _clean_extracted_text(text: str) -> str:
    """Çıkarımdan sonra bozuk karakterleri ve kontrol kodlarını temizle."""
    if not text:
        return ""
    # Replacement char ve PUA → kaldır (tamamen okunamayan)
    text = _PUA_RE.sub("", text)
    # Kontrol karakterleri (newline/tab hariç) → kaldır
    text = _CTRL_RE.sub("", text)
    # Soft hyphen (­) — bazı PDF'lerin satır sonu süslemesi, kelime ortasında görünür
    text = text.replace("­", "")
    # Çift boşlukları teklerle değiştir, NBSP'leri normal boşluğa çevir
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    return text


def _extract_page_text(page) -> str:
    """Bir sayfa için birden fazla extraction yöntemi dener, en sağlamını seçer.

    Bazı PDF'lerde font'un toUnicode CMap'i bozuk olur ve karakterler ?
    ya da PUA glif'i olarak gelir. Farklı `get_text` modları farklı sonuç
    verebilir — en az bozuk olanı tercih ediyoruz.
    """
    candidates: List[str] = []

    # Yöntem 1: standart text — çoğu PDF için yeterli, hızlı.
    try:
        t = page.get_text("text", sort=True) or ""
        candidates.append(t)
    except Exception:
        pass

    # Yöntem 2: dict — span-bazlı, bazen toUnicode bypass ediliyor.
    try:
        td = page.get_text("dict")
        parts: List[str] = []
        for block in td.get("blocks", []):
            for line in block.get("lines", []):
                spans = [s.get("text", "") for s in line.get("spans", [])]
                if spans:
                    parts.append("".join(spans))
        if parts:
            candidates.append("\n".join(parts))
    except Exception:
        pass

    # Yöntem 3: rawdict — daha ham, bazı font tablolarında daha iyi.
    try:
        t = page.get_text("blocks") or []
        parts = [b[4] for b in t if len(b) > 4 and isinstance(b[4], str)]
        if parts:
            candidates.append("\n".join(parts))
    except Exception:
        pass

    if not candidates:
        return ""

    # En az bozuk karakter oranına sahip aday + en uzun olanı dengeleyerek seç.
    # (Pure ratio en kısa boş string'i seçebilir, uzunluğa da bakıyoruz.)
    best = min(
        candidates,
        key=lambda x: (_garbage_ratio(x), -len(x.strip())),
    )
    return _clean_extracted_text(best).strip()


def extract_pdf_chunks(file_bytes: bytes) -> List[Dict]:
    if len(file_bytes) > MAX_PDF_BYTES:
        raise PDFLoadError("err_too_large", limit=MAX_PDF_BYTES // (1024 * 1024))

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFLoadError("err_open_failed", detail=str(exc)) from exc

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise PDFLoadError("err_encrypted")

        if doc.page_count > MAX_PDF_PAGES:
            raise PDFLoadError("err_too_many_pages", limit=MAX_PDF_PAGES)

        chunks: List[Dict] = []
        total_chars = 0
        total_garbage = 0
        for index, page in enumerate(doc):
            try:
                text = _extract_page_text(page)
            except Exception:
                text = ""
            if text:
                garbage = sum(
                    1 for ch in text
                    if _PUA_RE.match(ch) or ch == "?" or _CTRL_RE.match(ch)
                )
                total_chars += len(text)
                total_garbage += garbage
                chunks.append({"page": index + 1, "text": text})
    finally:
        doc.close()

    if not chunks:
        raise PDFLoadError("err_empty")

    # Yüksek garbage ratio'da kullanıcıyı uyar — ama yine de işle.
    if total_chars > 0 and total_garbage / total_chars > _GARBAGE_THRESHOLD:
        try:
            st.warning(t("warn_low_quality_pdf"))
        except Exception:
            # Test stub'unda warning olmayabilir, sessizce geç.
            pass

    return chunks


LLM_ERROR_PREFIX = "__LLM_ERROR__::"


def _extract_wait_seconds(msg: str) -> float:
    """Groq hata mesajından 'try again in Xs' değerini çek."""
    m = re.search(r"try again in ([\d.]+)s", msg, re.IGNORECASE)
    if m:
        try:
            return min(MAX_SINGLE_WAIT_SEC, float(m.group(1)) + 0.5)
        except ValueError:
            pass
    m = re.search(r"in ([\d.]+)\s*ms", msg, re.IGNORECASE)
    if m:
        try:
            return min(MAX_SINGLE_WAIT_SEC, float(m.group(1)) / 1000.0 + 0.2)
        except ValueError:
            pass
    return 12.0


def _is_retryable_error(msg: str) -> bool:
    lowered = msg.lower()
    if any(token in lowered for token in (
        "rate_limit", "rate limit", "too large", "tokens per minute",
        "timeout", "timed out", "connection", "temporar", "service unavailable",
        "overloaded", "internal server",
    )):
        return True
    if any(code in msg for code in ("413", "429", "500", "502", "503", "504")):
        return True
    return False


def ask_llm(
    prompt: str,
    model_name: str,
    max_tokens: int = 2048,
    timeout_sec: float = LLM_REQUEST_TIMEOUT_SEC,
    max_total_retry_wait_sec: float = MAX_TOTAL_RETRY_WAIT_SEC,
) -> str:
    """Rate-limit / geçici hatalarla başa çıkan LLM çağrısı.

    Hata durumunda LLM_ERROR_PREFIX ile başlayan bir string döndürür;
    çağıran taraf bunu st.error olarak gösterir.
    """
    if client is None:
        return f"{LLM_ERROR_PREFIX}{t('no_api_key')}"

    last_error = ""
    total_waited = 0.0
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
                timeout=timeout_sec,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            msg = str(exc)
            last_error = msg
            if not _is_retryable_error(msg) or attempt == MAX_RETRIES - 1:
                break
            base_wait = _extract_wait_seconds(msg)
            jitter = random.uniform(0.3, 1.2)
            remaining = max_total_retry_wait_sec - total_waited
            if remaining <= 0.5:
                break
            wait = min(base_wait + jitter, remaining, MAX_SINGLE_WAIT_SEC)
            time.sleep(wait)
            total_waited += wait

    return f"{LLM_ERROR_PREFIX}{last_error or t('llm_failed')}"


def stream_llm(
    prompt: str,
    model_name: str,
    max_tokens: int = 2048,
    timeout_sec: float = LLM_REQUEST_TIMEOUT_SEC,
    max_total_retry_wait_sec: float = MAX_TOTAL_RETRY_WAIT_SEC,
):
    """Streaming çağrısı; rate-limit'e takılırsa bir kez bekler ve yeniden dener.

    Üretilen her parça (token) yield edilir. Hata olursa LLM_ERROR_PREFIX'li
    tek bir string yield edip biter (caller bunu UI'da gösterir).
    """
    if client is None:
        yield f"{LLM_ERROR_PREFIX}{t('no_api_key')}"
        return

    total_waited = 0.0
    last_error = ""
    emitted_any = False
    for attempt in range(MAX_RETRIES):
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
                stream=True,
                timeout=timeout_sec,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content
                except (AttributeError, IndexError):
                    delta = None
                if delta:
                    emitted_any = True
                    yield delta
            return
        except Exception as exc:
            msg = str(exc)
            last_error = msg
            if emitted_any:
                yield f"{LLM_ERROR_PREFIX}{last_error or t('llm_failed')}"
                return
            if not _is_retryable_error(msg) or attempt == MAX_RETRIES - 1:
                break
            base_wait = _extract_wait_seconds(msg)
            jitter = random.uniform(0.3, 1.2)
            remaining = max_total_retry_wait_sec - total_waited
            if remaining <= 0.5:
                break
            wait = min(base_wait + jitter, remaining, MAX_SINGLE_WAIT_SEC)
            time.sleep(wait)
            total_waited += wait

    yield f"{LLM_ERROR_PREFIX}{last_error or t('llm_failed')}"


def rechunk_text(chunks: List[Dict], target_chars: int = CHUNK_CHAR_LIMIT) -> List[Dict]:
    """Küçük sayfaları birleştir, büyük sayfaları böl. Her parça ~target_chars karakter."""
    out: List[Dict] = []
    buf = ""
    pages: List[int] = []

    def flush():
        nonlocal buf, pages
        if buf.strip():
            out.append({"pages": pages[:], "text": buf.strip()})
        buf = ""
        pages = []

    for c in chunks:
        text = c["text"]
        page = c["page"]

        # Büyük sayfa → parçala
        while len(text) > target_chars:
            flush()
            out.append({"pages": [page], "text": text[:target_chars]})
            text = text[target_chars:]

        # Kalan kısmı buffer'a ekle
        if buf and len(buf) + len(text) + 2 > target_chars:
            flush()
        buf = (buf + "\n\n" + text) if buf else text
        if page not in pages:
            pages.append(page)

    flush()
    return out


def _summary_prompt_templates() -> Dict[str, str]:
    """Aktif dile göre özet prompt şablonlarını döndürür.

    chunk: tek bir PDF parçası için ara özet
    final: ara özetleri tek bir bütünleşik özete birleştirme
    direct: tüm PDF tek geçişte özetlenecek kadar küçükse
    """
    if get_lang() == "tr":
        chunk_prompt_template = """
ÖNEMLİ: Yanıtını mutlaka Türkçe yaz. PDF başka bir dilde olsa bile çıktı Türkçe olmalı.

Sen bir PDF asistanısın.
Bu PDF bölümüne göre en önemli konuları ve terimleri ayrıntılı biçimde özetle.
Kaynak belirtme. Sadece özet yaz.
Bu çıktı, 5-6 sayfalık ders notu çıkarmaya yetecek kadar dolu olsun.

Kurallar:
- Yüzeysel yazma
- En önemli konuları açık ve düzenli anlat
- Kritik terimleri tek tek açıkla
- Gerekirse konuları başlıklara ayır
- Gereksiz tekrar yapma
- Sadece PDF içeriğine dayan
- Kaynak, sayfa numarası, alıntı veya referans verme

Metin:
{text}
""".strip()

        final_prompt_template = """
ÖNEMLİ: Yanıtını mutlaka Türkçe yaz. PDF başka bir dilde olsa bile çıktı Türkçe olmalı.

Sen bir PDF asistanısın.
Aşağıda PDF'in farklı parçalarından çıkarılmış ara özetler var.
Bunların tamamını kullanarak tek bir büyük ve düzenli özet oluştur.

İstenen çıktı:
- PDF'in en önemli konularını açıkla
- En kritik terimleri açıkla
- Konuları mantıklı başlıklar altında topla
- Ders notu çıkarabilecek kadar dolu olsun
- 5-6 sayfalık not yazmaya yetecek kadar detaylı olsun
- Kaynak belirtme ve sayfa numarası verme
- Sadece özet yaz
- Çıktının sonunda "Kapanış" başlığı altında ana çıkarımları tamamla
- Numaralı veya maddeli liste başlatırsan tüm maddeleri eksiksiz bitir

Kurallar:
- Yüzeysel olma
- Gereksiz tekrar yapma
- Anlaşılır ve düzenli yaz
- Başlıklar ve alt başlıklar kullan
- Sadece verilen içerikten yararlan
- Yanıtı yarım bırakma; son cümle tam ve noktalı bitsin

Ara özetler:
{combined}
""".strip()

        direct_prompt_template = """
ÖNEMLİ: Yanıtını mutlaka Türkçe yaz. PDF başka bir dilde olsa bile çıktı Türkçe olmalı.

Sen bir PDF asistanısın.
Aşağıdaki PDF metninin tamamını kullanarak hızlı ama kapsamlı bir ders notu özeti oluştur.

İstenen çıktı:
- PDF'in ana konularını mantıklı başlıklarla düzenle
- Kritik terimleri açıkla
- Önemli maddeleri numaralı veya maddeli listelerle ver
- Gereksiz tekrar yapma
- Kaynak, sayfa numarası, alıntı veya referans verme
- En sonda "Kapanış" başlığıyla ana çıkarımları tamamla
- Yanıtı yarım bırakma; son cümle tam ve noktalı bitsin

PDF metni:
{text}
""".strip()
    else:
        chunk_prompt_template = """
IMPORTANT: Always respond in English. The output must be in English even if the PDF is in another language.

You are a PDF assistant.
Summarize the most important topics and terms from this PDF section.
Do not cite sources. Only provide a summary.
The output should be detailed enough to produce 5-6 pages of study notes.

Rules:
- Do not be superficial
- Explain the most important topics clearly
- Explain important terms one by one
- Split content into headings when needed
- Avoid unnecessary repetition
- Use only the PDF content
- Do not include page numbers or references

Text:
{text}
""".strip()

        final_prompt_template = """
IMPORTANT: Always respond in English. The output must be in English even if the PDF is in another language.

You are a PDF assistant.
Below are partial summaries from different parts of the PDF.
Create one large, organized summary from all of them.

Requirements:
- Explain the most important topics
- Explain critical terms
- Group topics under logical headings
- Be detailed enough for 5-6 pages of notes
- Do not cite sources or page numbers
- Return only the summary
- End with a "Closing Notes" section that completes the main takeaways
- If you start a numbered or bulleted list, finish every item

Rules:
- Do not be superficial
- Avoid unnecessary repetition
- Write clearly and in an organized way
- Use headings and subheadings
- Use only the provided content
- Do not leave the response unfinished; end with a complete sentence

Partial summaries:
{combined}
""".strip()

        direct_prompt_template = """
IMPORTANT: Always respond in English. The output must be in English even if the PDF is in another language.

You are a PDF assistant.
Use the full PDF text below to create a fast but comprehensive study-note summary.

Requirements:
- Organize the main topics under clear headings
- Explain critical terms
- Use numbered or bulleted lists for important points
- Avoid unnecessary repetition
- Do not cite sources, page numbers, quotes, or references
- End with a "Closing Notes" heading and complete the main takeaways
- Do not leave the response unfinished; end with a complete sentence

PDF text:
{text}
""".strip()

    return {
        "chunk": chunk_prompt_template,
        "final": final_prompt_template,
        "direct": direct_prompt_template,
    }


def _summarize_chunks_in_parallel(
    pieces: List[Dict],
    chunk_prompt_template: str,
    model_name: str,
) -> tuple[List[str], int]:
    """Parça özetlerini paralel olarak üretir; sıra korunur.

    Sıralı çalıştırma yerine ThreadPoolExecutor ile aynı anda en fazla
    MAX_PARALLEL_CHUNK_REQUESTS çağrı yapılır. Bu, çok parçalı PDF'lerde
    özet süresini doğrudan parça sayısına bölünene kadar düşürür.

    Returns:
        (partial_summaries, failure_count) — başarısızlar listede yer almaz.
    """
    workers = max(1, min(MAX_PARALLEL_CHUNK_REQUESTS, len(pieces)))
    results: List[Optional[str]] = [None] * len(pieces)

    def _do(index: int) -> None:
        prompt = chunk_prompt_template.format(text=pieces[index]["text"])
        results[index] = ask_llm(prompt, model_name, max_tokens=SUMMARY_CHUNK_MAX_TOKENS)

    progress = st.progress(0.0)
    completed = 0

    if workers <= 1:
        # Test/güvenli mod — sıralı.
        for index in range(len(pieces)):
            _do(index)
            completed += 1
            progress.progress(completed / len(pieces))
            if REQUEST_PACING_SEC > 0 and index < len(pieces) - 1:
                time.sleep(REQUEST_PACING_SEC)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_do, i): i for i in range(len(pieces))}
            for future in as_completed(futures):
                # Hatayı yutma — tek bir parça patlasa diğerleri devam etsin.
                try:
                    future.result()
                except Exception as exc:  # pragma: no cover - defansif
                    failed_index = futures[future]
                    results[failed_index] = f"{LLM_ERROR_PREFIX}{exc}"
                completed += 1
                progress.progress(completed / len(pieces))

    progress.empty()

    partial_summaries: List[str] = []
    failures = 0
    for result in results:
        if result and not result.startswith(LLM_ERROR_PREFIX):
            partial_summaries.append(result)
        else:
            failures += 1
    return partial_summaries, failures


def _merge_summary_groups_if_needed(
    partial_summaries: List[str],
    final_prompt_template: str,
    model_name: str,
    group_size: int = 5,
) -> List[str]:
    """Çok sayıda ara özeti hiyerarşik olarak birleştirir."""

    def combine_group(texts: List[str]) -> Optional[str]:
        joined = "\n\n---\n\n".join(texts)
        result = ask_llm(
            final_prompt_template.format(combined=joined),
            model_name,
            max_tokens=SUMMARY_GROUP_MAX_TOKENS,
        )
        if result.startswith(LLM_ERROR_PREFIX):
            return None
        return result

    while len(partial_summaries) > group_size:
        merged: List[str] = []
        for i in range(0, len(partial_summaries), group_size):
            group = partial_summaries[i:i + group_size]
            combined_chunk = combine_group(group)
            if combined_chunk:
                merged.append(combined_chunk)
            else:
                merged.extend(group)
        partial_summaries = merged
    return partial_summaries


def summarize_pdf(chunks: List[Dict], model_name: str) -> str:
    templates = _summary_prompt_templates()
    chunk_prompt_template = templates["chunk"]
    final_prompt_template = templates["final"]
    direct_prompt_template = templates["direct"]

    total_chars = sum(len(chunk["text"]) for chunk in chunks)
    if total_chars <= DIRECT_SUMMARY_CHAR_LIMIT:
        context = build_context(
            chunks,
            question="",
            max_chars=DIRECT_SUMMARY_CHAR_LIMIT + 1000,
        )
        result = ask_llm(
            direct_prompt_template.format(text=context),
            model_name,
            max_tokens=SUMMARY_FINAL_MAX_TOKENS,
        )
        if result.startswith(LLM_ERROR_PREFIX):
            if not _is_retryable_error(result):
                return result
        else:
            return _complete_summary_if_needed(result, context, model_name)

    # 1) PDF'i daha küçük, rate-limit dostu parçalara böl
    pieces = rechunk_text(chunks, target_chars=CHUNK_CHAR_LIMIT)

    # 2) Her parça için ara özet — paralel olarak (eski sıralı döngüye göre
    #    parça sayısına yakın oranda hızlanma sağlar).
    partial_summaries, failures = _summarize_chunks_in_parallel(
        pieces, chunk_prompt_template, model_name
    )

    if not partial_summaries:
        return f"{LLM_ERROR_PREFIX}{t('llm_failed')}"

    if failures:
        st.warning(t("llm_partial"))

    # 3) Ara özetler çoksa, önce grup halinde birleştir (hiyerarşik)
    partial_summaries = _merge_summary_groups_if_needed(
        partial_summaries, final_prompt_template, model_name
    )

    combined = "\n\n---\n\n".join(partial_summaries)
    # 4) Son büyük birleştirme
    if len(partial_summaries) == 1:
        return _complete_summary_if_needed(partial_summaries[0], combined, model_name)

    final_prompt = final_prompt_template.format(combined=combined)
    final_result = ask_llm(final_prompt, model_name, max_tokens=SUMMARY_FINAL_MAX_TOKENS)
    if final_result.startswith(LLM_ERROR_PREFIX):
        # Final çağrısı patlarsa, en azından ara özetleri birleştirip ver
        return "\n\n---\n\n".join(partial_summaries)
    return _complete_summary_if_needed(final_result, combined, model_name)


def stream_summarize_pdf(chunks: List[Dict], model_name: str):
    """Özetin son geçişini streaming olarak yield eder.

    Doğrudan yol (küçük PDF): tek bir streaming çağrısıyla anında token akışı.
    Parçalı yol: ara özetler önce paralel olarak (sessizce) üretilir, sonra
    birleştirme adımı streaming olarak yield edilir. Kullanıcı tüm cevabı
    beklemek yerine ilk token'ları saniyeler içinde görür.

    Hata olursa LLM_ERROR_PREFIX'li tek bir parça yield edip biter.
    """
    templates = _summary_prompt_templates()
    chunk_prompt_template = templates["chunk"]
    final_prompt_template = templates["final"]
    direct_prompt_template = templates["direct"]

    total_chars = sum(len(chunk["text"]) for chunk in chunks)
    if total_chars <= DIRECT_SUMMARY_CHAR_LIMIT:
        context = build_context(
            chunks,
            question="",
            max_chars=DIRECT_SUMMARY_CHAR_LIMIT + 1000,
        )
        yield from stream_llm(
            direct_prompt_template.format(text=context),
            model_name,
            max_tokens=SUMMARY_FINAL_MAX_TOKENS,
        )
        return

    pieces = rechunk_text(chunks, target_chars=CHUNK_CHAR_LIMIT)
    partial_summaries, failures = _summarize_chunks_in_parallel(
        pieces, chunk_prompt_template, model_name
    )

    if not partial_summaries:
        yield f"{LLM_ERROR_PREFIX}{t('llm_failed')}"
        return

    if failures:
        st.warning(t("llm_partial"))

    partial_summaries = _merge_summary_groups_if_needed(
        partial_summaries, final_prompt_template, model_name
    )

    if len(partial_summaries) == 1:
        # Tek bir ara özet kaldı — onu olduğu gibi tek parça olarak akıt.
        yield partial_summaries[0]
        return

    combined = "\n\n---\n\n".join(partial_summaries)
    yield from stream_llm(
        final_prompt_template.format(combined=combined),
        model_name,
        max_tokens=SUMMARY_FINAL_MAX_TOKENS,
    )


def _looks_incomplete_summary(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if re.search(r"(\b\d+\.?|[,;:–—-])\s*$", stripped):
        return True
    if re.search(
        r"\b(ve|veya|ile|ama|ancak|çünkü|olarak|the|and|or|but|because|with|as)\s*$",
        stripped,
        re.IGNORECASE,
    ):
        return True
    return False


def _clip_for_continuation(text: str, limit: int = 24000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n\n[...]\n\n" + text[-half:]


def _build_summary_continuation_prompt(summary: str, source_notes: str) -> str:
    source = _clip_for_continuation(source_notes)
    tail = summary[-5000:]
    if get_lang() == "tr":
        return f"""
ÖNEMLİ: Yanıtını mutlaka Türkçe yaz.

Bir PDF özetinin sonu yarım kalmış görünüyor.
Ara özetleri ve mevcut özetin son kısmını kullanarak sadece eksik kalan devam metnini yaz.

Kurallar:
- Önceden yazılmış bölümleri tekrar etme
- Kaldığı yerden doğal biçimde devam et
- Eksik kalan maddeleri tamamla
- En sonda "Kapanış" başlığıyla ana çıkarımları bitir
- Cevabı tam bir cümleyle ve noktayla kapat

Ara özetler:
{source}

Mevcut özetin son kısmı:
{tail}
""".strip()

    return f"""
IMPORTANT: Always respond in English.

The end of a PDF summary appears to be cut off.
Using the partial notes and the tail of the current summary, write only the missing continuation.

Rules:
- Do not repeat sections that are already written
- Continue naturally from where it stopped
- Complete any unfinished lists
- End with a "Closing Notes" heading and finish the main takeaways
- Close with a complete sentence

Partial notes:
{source}

Tail of the current summary:
{tail}
""".strip()


def _complete_summary_if_needed(summary: str, source_notes: str, model_name: str) -> str:
    if not _looks_incomplete_summary(summary):
        return summary

    continuation = ask_llm(
        _build_summary_continuation_prompt(summary, source_notes),
        model_name,
        max_tokens=SUMMARY_CONTINUATION_MAX_TOKENS,
    )
    if continuation.startswith(LLM_ERROR_PREFIX) or not continuation.strip():
        return summary
    return summary.rstrip() + "\n\n" + continuation.strip()


SUMMARY_PDF_FONTS = {
    "body": "MTKSummaryBody",
    "body_bold": "MTKSummaryBodyBold",
    "heading": "MTKSummaryHeading",
    "heading_bold": "MTKSummaryHeadingBold",
}

SUMMARY_PDF_COLORS = {
    "bg": "#0A0A0B",
    "panel": "#111113",
    "panel_soft": "#17181A",
    "grid": "#202124",
    "text": "#E8E6E1",
    "text_dim": "#AAA69F",
    "accent": "#C9F06B",
    "accent_2": "#6BC9F0",
    "border": "#333438",
}


def _find_first_existing_path(paths: List[str]) -> Optional[str]:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _font_candidates(*names: str, bold: bool = False, serif: bool = False) -> List[str]:
    """Sıralı font arama yolları. Linux'ta DejaVu/Liberation gibi Türkçe
    karakterleri (ı, ğ, ş) destekleyen fontları tercih ediyoruz — ReportLab'in
    yerleşik Helvetica/Times'ı Latin Extended-A desteklemez, kareler oluşur.

    Args:
        names: Windows font dosya adları (öncelik)
        bold: True ise bold varyantları tercih et
        serif: True ise serif varyantları tercih et (heading'ler için)
    """
    windir = os.environ.get("WINDIR", r"C:\Windows")
    win_fonts = os.path.join(windir, "Fonts")
    paths = [os.path.join(win_fonts, name) for name in names]

    # Linux paths — packages.txt fonts-dejavu-core fonts-dejavu-extra ile gelir
    linux_paths = []
    if serif:
        if bold:
            linux_paths.extend([
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            ])
        else:
            linux_paths.extend([
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            ])
    else:
        if bold:
            linux_paths.extend([
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            ])
        else:
            linux_paths.extend([
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ])
    paths.extend(linux_paths)

    # macOS paths
    paths.extend([
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Georgia.ttf",
    ])

    # Repo'ya bundled fontlar (varsa) — son çare
    here = os.path.dirname(os.path.abspath(__file__))
    paths.extend([
        os.path.join(here, "fonts", "DejaVuSans.ttf"),
        os.path.join(here, "fonts", "DejaVuSans-Bold.ttf"),
    ])

    return paths


def _register_font_if_available(font_name: str, paths: List[str], fallback: str) -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_file = _find_first_existing_path(paths)
    if not font_file:
        return fallback

    try:
        pdfmetrics.getFont(font_name)
    except KeyError:
        pdfmetrics.registerFont(TTFont(font_name, font_file))
    return font_name


def _register_summary_pdf_fonts() -> Dict[str, str]:
    from reportlab.pdfbase import pdfmetrics

    fonts = {
        "body": _register_font_if_available(
            SUMMARY_PDF_FONTS["body"],
            _font_candidates("segoeui.ttf", "arial.ttf", "calibri.ttf", bold=False, serif=False),
            "Helvetica",
        ),
        "body_bold": _register_font_if_available(
            SUMMARY_PDF_FONTS["body_bold"],
            _font_candidates("segoeuib.ttf", "arialbd.ttf", "calibrib.ttf", bold=True, serif=False),
            "Helvetica-Bold",
        ),
        "heading": _register_font_if_available(
            SUMMARY_PDF_FONTS["heading"],
            _font_candidates("georgia.ttf", "cambria.ttc", "segoeui.ttf", bold=False, serif=True),
            "Times-Roman",
        ),
        "heading_bold": _register_font_if_available(
            SUMMARY_PDF_FONTS["heading_bold"],
            _font_candidates("georgiab.ttf", "cambriab.ttf", "segoeuib.ttf", "arialbd.ttf", bold=True, serif=True),
            "Times-Bold",
        ),
    }
    pdfmetrics.registerFontFamily(
        fonts["body"],
        normal=fonts["body"],
        bold=fonts["body_bold"],
        italic=fonts["body"],
        boldItalic=fonts["body_bold"],
    )
    return fonts


def _strip_summary_markdown(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", text.strip())
    text = re.sub(r"^\s*[-*•]\s+", "", text)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text.strip()


def _is_summary_heading_line(line: str) -> bool:
    stripped = line.strip()
    if re.match(r"^#{1,6}\s+", stripped):
        return True
    if re.match(r"^\s*([-*•]|\d+[.)])\s+", stripped):
        return False
    if len(stripped) > 96 or len(stripped.split()) > 9:
        return False
    if re.search(r"[.!?;,]\s*$", stripped):
        return False
    if stripped.endswith(":") and len(stripped.split()) > 5:
        return False
    return True


def _normalize_summary_paragraph(lines: List[str]) -> str:
    cleaned = [_strip_summary_markdown(line) for line in lines if line.strip()]
    return " ".join(cleaned).strip()


def _summary_pdf_filename(file_name: Optional[str]) -> str:
    base = os.path.splitext(file_name or "pdf-summary")[0]
    safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._-")
    suffix = "ozet" if get_lang() == "tr" else "summary"
    return f"{safe_base or 'pdf'}-{suffix}.pdf"


def create_summary_pdf_bytes(summary: str, source_file_name: Optional[str] = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    fonts = _register_summary_pdf_fonts()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=24 * mm,
        rightMargin=22 * mm,
        topMargin=36 * mm,
        bottomMargin=20 * mm,
        title=t("summary_pdf_title"),
        author="mtk. PDF Assistant",
    )

    styles = getSampleStyleSheet()
    colors_map = {key: colors.HexColor(value) for key, value in SUMMARY_PDF_COLORS.items()}

    def draw_page(canvas: Canvas, built_doc) -> None:
        width, height = A4
        left = built_doc.leftMargin
        right = width - built_doc.rightMargin

        canvas.saveState()
        canvas.setFillColor(colors_map["bg"])
        canvas.rect(0, 0, width, height, stroke=0, fill=1)

        canvas.setStrokeColor(colors_map["grid"])
        canvas.setLineWidth(0.18)
        step = 18 * mm
        x = 0
        while x <= width:
            canvas.line(x, 0, x, height)
            x += step
        y = 0
        while y <= height:
            canvas.line(0, y, width, y)
            y += step

        canvas.setFillColor(colors_map["panel"])
        canvas.roundRect(left - 8 * mm, height - 30 * mm, right - left + 16 * mm, 18 * mm, 8, stroke=0, fill=1)
        canvas.setStrokeColor(colors_map["border"])
        canvas.setLineWidth(0.5)
        canvas.roundRect(left - 8 * mm, height - 30 * mm, right - left + 16 * mm, 18 * mm, 8, stroke=1, fill=0)

        brand_y = height - 20 * mm
        canvas.setFont(fonts["heading_bold"], 15)
        canvas.setFillColor(colors_map["text"])
        canvas.drawString(left, brand_y, "mtk")
        brand_width = canvas.stringWidth("mtk", fonts["heading_bold"], 15)
        canvas.setFillColor(colors_map["accent"])
        canvas.drawString(left + brand_width, brand_y, ".")

        tag = "PDF ASSISTANT"
        canvas.setFont(fonts["body_bold"], 7.5)
        tag_width = canvas.stringWidth(tag, fonts["body_bold"], 7.5)
        tag_x = right - tag_width - 11 * mm
        canvas.setFillColor(colors_map["accent"])
        canvas.circle(tag_x - 5, brand_y + 3, 2.1, stroke=0, fill=1)
        canvas.setFillColor(colors_map["text"])
        canvas.drawString(tag_x, brand_y, tag)

        canvas.setStrokeColor(colors_map["accent"])
        canvas.setLineWidth(1.2)
        canvas.line(left, height - 34 * mm, left + 24 * mm, height - 34 * mm)

        canvas.setStrokeColor(colors_map["border"])
        canvas.setLineWidth(0.5)
        canvas.line(left, 15 * mm, right, 15 * mm)
        canvas.setFont(fonts["body"], 8)
        canvas.setFillColor(colors_map["text_dim"])
        canvas.drawString(left, 10 * mm, "mtk. PDF Assistant")
        canvas.drawRightString(right, 10 * mm, str(canvas.getPageNumber()))
        canvas.restoreState()

    title_style = ParagraphStyle(
        "SummaryTitle",
        parent=styles["Title"],
        fontName=fonts["heading_bold"],
        fontSize=36,
        leading=41,
        alignment=0,
        textColor=colors_map["accent"],
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "SummaryMeta",
        parent=styles["BodyText"],
        fontName=fonts["body"],
        fontSize=8.5,
        leading=12,
        textColor=colors_map["text_dim"],
        spaceAfter=18,
    )
    heading_style = ParagraphStyle(
        "SummaryHeading",
        parent=styles["Heading2"],
        fontName=fonts["heading_bold"],
        fontSize=25,
        leading=30,
        textColor=colors_map["text"],
        spaceBefore=18,
        spaceAfter=9,
    )
    body_style = ParagraphStyle(
        "SummaryBody",
        parent=styles["BodyText"],
        fontName=fonts["body"],
        fontSize=11,
        leading=17,
        textColor=colors_map["text"],
        spaceAfter=10,
    )
    bullet_style = ParagraphStyle(
        "SummaryBullet",
        parent=body_style,
        leftIndent=18,
        firstLineIndent=-12,
        spaceBefore=2,
        spaceAfter=7,
    )
    number_style = ParagraphStyle(
        "SummaryNumber",
        parent=body_style,
        leftIndent=23,
        firstLineIndent=-18,
        spaceBefore=2,
        spaceAfter=7,
    )

    story = [
        Paragraph(escape(t("summary_pdf_title")), title_style),
    ]
    if source_file_name:
        story.append(Paragraph(f"PDF: {escape(source_file_name)}", meta_style))

    has_summary_content = False

    def append_paragraph(buffered_lines: List[str]) -> None:
        nonlocal has_summary_content
        paragraph = _normalize_summary_paragraph(buffered_lines)
        if paragraph:
            story.append(Paragraph(escape(paragraph), body_style))
            has_summary_content = True

    blocks = re.split(r"\n\s*\n", (summary or "").strip())
    for block in blocks:
        paragraph_buffer: List[str] = []
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            numbered_match = re.match(r"^\s*(\d+)[.)]\s+(.*)", line)
            bullet_match = re.match(r"^\s*[-*•]\s+(.*)", line)

            if _is_summary_heading_line(line):
                append_paragraph(paragraph_buffer)
                paragraph_buffer = []
                story.append(Paragraph(escape(_strip_summary_markdown(line)), heading_style))
                has_summary_content = True
                continue

            if numbered_match:
                append_paragraph(paragraph_buffer)
                paragraph_buffer = []
                number = numbered_match.group(1)
                item = escape(_strip_summary_markdown(numbered_match.group(2)))
                story.append(
                    Paragraph(
                        f'<font name="{fonts["body_bold"]}" color="{SUMMARY_PDF_COLORS["accent"]}">{number}.</font> {item}',
                        number_style,
                    )
                )
                has_summary_content = True
                continue

            if bullet_match:
                append_paragraph(paragraph_buffer)
                paragraph_buffer = []
                item = escape(_strip_summary_markdown(bullet_match.group(1)))
                story.append(
                    Paragraph(
                        f'<font name="{fonts["body_bold"]}" color="{SUMMARY_PDF_COLORS["accent"]}">•</font> {item}',
                        bullet_style,
                    )
                )
                has_summary_content = True
                continue

            paragraph_buffer.append(line)

        append_paragraph(paragraph_buffer)

    if not has_summary_content:
        story.append(Paragraph(escape(t("llm_empty")), body_style))

    story.append(Spacer(1, 4 * mm))
    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return buffer.getvalue()


def clean_text_for_tts(text: str) -> str:
    """Web Speech API'sine verilmeden önce markdown işaretlerini ve görsel
    süslemeleri kaldırır — kullanıcı "iki yıldız" duymak istemez.

    Korunanlar: cümle yapısı, noktalama, gerçek kelimeler.
    """
    if not text:
        return ""
    # Bold/italic işaretleri
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\1", text)
    # Inline code backticks
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Markdown başlıkları
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Liste işaretleri
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    # Çoklu boş satırları sadeleştir
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_tts_widget(summary: str) -> None:
    """Tarayıcının yerleşik speechSynthesis API'si ile özeti sesli okuyan widget.

    Tamamen tarayıcıda çalışır — API çağrısı, gecikme, maliyet yok. Türkçe ve
    İngilizce için sistemde mevcut ses motorunu (genellikle Microsoft/Apple
    yerleşik sesleri) kullanır.

    Aynı sebepten test edilmesi pratik değil — bu widget HTML/JS injection'dur.
    """
    if not summary:
        return

    cleaned = clean_text_for_tts(summary)
    lang = get_lang()
    voice_lang = "tr-TR" if lang == "tr" else "en-US"
    text_js = json.dumps(cleaned)

    label_play = t("tts_play")
    label_pause = t("tts_pause")
    label_resume = t("tts_resume")
    label_stop = t("tts_stop")
    label_speaking = t("tts_speaking")
    label_paused = t("tts_paused")
    label_error = t("tts_error")
    label_speed = t("tts_speed")

    # Web Speech API'de rate aralığı 0.1 - 10. Bu seti kullanıcı istedi.
    # 3x üstündeki hızlar bazı tarayıcılarda kalitesizleşebilir, ama desteklenir.
    speed_options = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0, 10.0]
    speed_options_html = "".join(
        f'<option value="{v}"{" selected" if v == 1.0 else ""}>{v}x</option>'
        for v in speed_options
    )

    html = fr"""
<style>
  .tts {{
    display: flex; gap: 10px; align-items: center;
    flex-wrap: wrap;
    font-family: 'DM Sans', system-ui, sans-serif;
    color: #e8e6e1;
    padding: 6px 0;
  }}
  @media (max-width: 640px) {{
    .tts {{ gap: 8px; }}
    .tts button {{ padding: 8px 12px !important; font-size: 12.5px; }}
    .tts .speed-group {{
      width: 100%;
      margin-left: 0;
      padding-left: 0;
      border-left: none;
      border-top: 1px solid #2a2a2e;
      padding-top: 8px;
    }}
    .tts select {{ flex: 1; }}
  }}
  .tts button {{
    background: #1a1a1e;
    color: #e8e6e1;
    border: 1px solid #2a2a2e;
    padding: 9px 18px;
    border-radius: 12px;
    font-weight: 500;
    cursor: pointer;
    font-size: 13.5px;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .tts button:hover:not(:disabled) {{
    background: #2a2a2e;
    border-color: #c9f06b;
  }}
  .tts button.primary {{
    background: #c9f06b;
    color: #0a0a0b;
    border-color: #c9f06b;
    font-weight: 600;
  }}
  .tts button.primary:hover:not(:disabled) {{
    background: #d4f582;
  }}
  .tts button:disabled {{ opacity: 0.35; cursor: not-allowed; }}
  .tts .speed-group {{
    display: inline-flex; align-items: center; gap: 6px;
    margin-left: 4px;
    padding: 0 4px 0 12px;
    border-left: 1px solid #2a2a2e;
  }}
  .tts .speed-label {{ color: #8a8884; font-size: 12.5px; }}
  .tts select {{
    background: #1a1a1e;
    color: #e8e6e1;
    border: 1px solid #2a2a2e;
    border-radius: 10px;
    padding: 7px 10px;
    font-size: 13px;
    font-family: inherit;
    cursor: pointer;
    outline: none;
  }}
  .tts select:hover {{ border-color: #c9f06b; }}
  .tts select:focus {{ border-color: #c9f06b; }}
  .tts .status {{ color: #8a8884; font-size: 12.5px; margin-left: 4px; }}
  .tts .dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: #c9f06b; display: inline-block;
    animation: pulse 1.2s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 0.4; transform: scale(0.8); }}
    50%      {{ opacity: 1;   transform: scale(1); }}
  }}
</style>
<div class="tts">
  <button class="primary" id="ttsPlay" onclick="ttsPlay()">🔊 {label_play}</button>
  <button id="ttsPause" onclick="ttsPause()" disabled>⏸ {label_pause}</button>
  <button id="ttsStop" onclick="ttsStop()" disabled>⏹ {label_stop}</button>
  <span class="speed-group">
    <span class="speed-label">{label_speed}:</span>
    <select id="ttsRate" onchange="ttsRateChange()">
      {speed_options_html}
    </select>
  </span>
  <span class="status" id="ttsStatus"></span>
</div>
<script>
(function() {{
  const text = {text_js};
  const synth = window.speechSynthesis;

  // ─── State ─────────────────────────────────────────────────────────────
  // Metni cümlelere bölüp tek tek oynatıyoruz — bu sayede hız değişirse
  // mevcut cümle eski hızda tamamlanır, SONRAKİ cümle yeni hızda başlar.
  // Yeniden başlatma YOK.
  let chunks = [];
  let chunkIndex = 0;
  let isPlaying = false;
  let utter = null;

  const $play   = document.getElementById('ttsPlay');
  const $pause  = document.getElementById('ttsPause');
  const $stop   = document.getElementById('ttsStop');
  const $status = document.getElementById('ttsStatus');

  function chunkText(t) {{
    // Cümle sonlandırıcılar (. ! ? …) ardından gelen boşluğa göre böl.
    const sentences = t.split(/(?<=[.!?…])\s+/);
    const out = [];
    for (const s of sentences) {{
      const trimmed = s.trim();
      if (!trimmed) continue;
      // Sert satır kırılmalarını da koru (listeler, başlıklar).
      for (const line of trimmed.split(/\n+/)) {{
        const ln = line.trim();
        if (!ln) continue;
        // Çok uzun cümleler için (virgül/noktalı virgül vs.) clause boundary'lerde böl.
        if (ln.length > 250) {{
          for (const p of ln.split(/(?<=[,;:])\s+/)) {{
            const pp = p.trim();
            if (pp) out.push(pp);
          }}
        }} else {{
          out.push(ln);
        }}
      }}
    }}
    return out.length ? out : [t];
  }}

  function setStatus(msg, withDot) {{
    $status.innerHTML = withDot ? '<span class="dot"></span> ' + msg : msg;
  }}

  function resetUI() {{
    isPlaying = false;
    chunkIndex = 0;
    $play.disabled = false;
    $play.textContent = '🔊 {label_play}';
    $pause.disabled = true;
    $stop.disabled = true;
    setStatus('', false);
  }}

  function currentRate() {{
    const el = document.getElementById('ttsRate');
    const v = parseFloat(el ? el.value : '1');
    // Web Speech API rate aralığı: 0.1 - 10
    return Math.max(0.1, Math.min(10, isFinite(v) ? v : 1));
  }}

  function speakNextChunk() {{
    if (!isPlaying || chunkIndex >= chunks.length) {{
      resetUI();
      return;
    }}
    utter = new SpeechSynthesisUtterance(chunks[chunkIndex]);
    utter.lang = '{voice_lang}';
    utter.rate = currentRate();         // <- her yeni cümle için anlık rate okunur
    utter.pitch = 1.0;
    utter.onend = function() {{
      if (!isPlaying) return;            // stop edilmişse ilerleme
      chunkIndex++;
      speakNextChunk();
    }};
    utter.onerror = function(e) {{
      // 'canceled' / 'interrupted' — biz tetikledik, sessiz geç.
      if (e.error === 'canceled' || e.error === 'interrupted') return;
      setStatus('{label_error}: ' + e.error, false);
      isPlaying = false;
      resetUI();
    }};
    synth.speak(utter);
  }}

  window.ttsPlay = function() {{
    // Duraklatılmışsa: kaldığı yerden devam et (mevcut cümle eski hızda
    // biter; sonraki cümle yeni hızda başlar).
    if (synth.paused && isPlaying) {{
      synth.resume();
      $play.textContent = '🔊 {label_play}';
      $pause.disabled = false;
      setStatus('{label_speaking}...', true);
      return;
    }}
    if (isPlaying) return;

    synth.cancel();
    chunks = chunkText(text);
    chunkIndex = 0;
    isPlaying = true;
    $play.disabled = true;
    $pause.disabled = false;
    $stop.disabled = false;
    setStatus('{label_speaking}...', true);
    // Tarayıcının cancel sonrası state'i temizlemesi için minik gecikme.
    setTimeout(speakNextChunk, 50);
  }};

  window.ttsPause = function() {{
    if (synth.speaking && !synth.paused) {{
      synth.pause();
      $play.disabled = false;
      $play.textContent = '▶ {label_resume}';
      $pause.disabled = true;
      setStatus('{label_paused}', false);
    }}
  }};

  window.ttsStop = function() {{
    isPlaying = false;
    synth.cancel();
    resetUI();
  }};

  window.ttsRateChange = function() {{
    // ÖNEMLİ: hiçbir şey yapma. Mevcut cümle eski hızda tamamlanır,
    // SONRAKİ cümle (speakNextChunk içindeki currentRate() çağrısı sayesinde)
    // yeni hızda başlar. Hız değişimi → restart YOK.
  }};

  // Cleanup: sekme kapanırken ses kalmasın
  window.addEventListener('beforeunload', () => synth.cancel());
}})();
</script>
"""
    components.html(html, height=72)


_WORD_RE = re.compile(r"\w{3,}", re.UNICODE)
_TR_STOPWORDS = {
    "için", "veya", "değil", "olarak", "daha", "bir", "bu", "şu", "ile",
    "the", "and", "for", "with", "that", "this", "from", "are", "was",
    "were", "have", "has", "you", "your", "what", "which", "when", "where",
}


def _score_tokens(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _TR_STOPWORDS}


def build_context(chunks: List[Dict], question: str = "", max_chars: int = 16000) -> str:
    """Soru verilirse alakalılık skoruna göre seçer; yoksa baştan alır.
    Seçimi sonra sayfa sırasına göre dizip mantıksal akışı korur."""
    if not chunks:
        return ""

    total_context_chars = sum(len(chunk["text"]) + 16 for chunk in chunks)
    if total_context_chars <= max_chars:
        ordered_chunks = sorted(chunks, key=lambda chunk: chunk["page"])
        return "\n\n".join(f"[Page {chunk['page']}]\n{chunk['text']}" for chunk in ordered_chunks)

    if not question.strip():
        selected = []
        total = 0
        for chunk in chunks:
            part = f"[Page {chunk['page']}]\n{chunk['text']}"
            if total and total + len(part) > max_chars:
                break
            selected.append(part)
            total += len(part)
        return "\n\n".join(selected)

    q_tokens = _score_tokens(question)
    scored = []
    for chunk in chunks:
        c_tokens = _score_tokens(chunk["text"])
        overlap = len(q_tokens & c_tokens) if q_tokens else 0
        scored.append((overlap, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]["page"]))

    picked: List[Dict] = []
    total = 0
    for _, chunk in scored:
        part = f"[Page {chunk['page']}]\n{chunk['text']}"
        if total and total + len(part) > max_chars:
            continue
        picked.append(chunk)
        total += len(part)
        if total >= max_chars:
            break

    if not picked:
        # Soru hiç eşleşmedi → baştan al, en azından bir şey ver
        return build_context(chunks, question="", max_chars=max_chars)

    picked.sort(key=lambda c: c["page"])
    return "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in picked)


def _build_chat_prompt(question: str, context: str, history: List[Dict]) -> str:
    recent = history[-3:] if history else []
    if get_lang() == "tr":
        history_block = ""
        if recent:
            lines = ["Önceki konuşma (referans için):"]
            for turn in recent:
                lines.append(f"S: {turn['q']}\nC: {turn['a']}")
            history_block = "\n".join(lines) + "\n\n"
        return (
            "ÖNEMLİ: Yanıtını mutlaka Türkçe yaz. PDF başka bir dilde olsa bile çıktı Türkçe olmalı.\n\n"
            "Sen bir PDF asistanısın. Aşağıdaki PDF içeriğine dayanarak kullanıcının sorusunu yanıtla.\n"
            "Sadece PDF içeriğine dayan. Bilmediğin şeyi uydurma; PDF'te yoksa açıkça söyle.\n"
            "Yanıtın açık, düzenli ve anlaşılır olsun.\n\n"
            f"{history_block}"
            f"PDF İçeriği:\n{context}\n\n"
            f"Soru: {question}"
        )
    history_block = ""
    if recent:
        lines = ["Previous conversation (for reference):"]
        for turn in recent:
            lines.append(f"Q: {turn['q']}\nA: {turn['a']}")
        history_block = "\n".join(lines) + "\n\n"
    return (
        "IMPORTANT: Always respond in English. The output must be in English even if the PDF is in another language.\n\n"
        "You are a PDF assistant. Answer the user's question based on the PDF content below.\n"
        "Only use the PDF content. Do not invent facts; say so if it isn't there.\n"
        "Your answer should be clear, organized, and easy to follow.\n\n"
        f"{history_block}"
        f"PDF Content:\n{context}\n\n"
        f"Question: {question}"
    )


def chat_with_pdf(question: str, chunks: List[Dict], model_name: str,
                  history: Optional[List[Dict]] = None) -> str:
    context = build_context(chunks, question=question, max_chars=CHAT_CONTEXT_CHAR_LIMIT)
    prompt = _build_chat_prompt(question, context, history or [])
    return ask_llm(prompt, model_name)


def chat_with_pdf_stream(question: str, chunks: List[Dict], model_name: str,
                         history: Optional[List[Dict]] = None):
    context = build_context(chunks, question=question, max_chars=CHAT_CONTEXT_CHAR_LIMIT)
    prompt = _build_chat_prompt(question, context, history or [])
    return stream_llm(prompt, model_name)


def inject_theme() -> None:
    css = r"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap');

            :root {
                --bg: #0a0a0b;
                --bg2: #111113;
                --bg3: #1a1a1e;
                --text: #e8e6e1;
                --text-dim: #8a8884;
                --accent: #c9f06b;
                --accent-soft: rgba(201, 240, 107, 0.14);
                --accent2: #6bc9f0;
                --accent3: #f06b8a;
                --border: #2a2a2e;
                --radius: 14px;
            }

            *, *::before, *::after { box-sizing: border-box; }

            ::selection { background: var(--accent); color: var(--bg); }

            html, body, [data-testid="stAppViewContainer"], .stApp {
                background: var(--bg);
                color: var(--text);
                font-family: 'DM Sans', sans-serif;
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(ellipse 60% 50% at 70% 30%, rgba(201, 240, 107, 0.06), transparent 60%),
                    radial-gradient(ellipse 40% 50% at 20% 60%, rgba(107, 201, 240, 0.04), transparent 60%),
                    linear-gradient(180deg, rgba(10, 10, 11, 1), rgba(10, 10, 11, 1));
            }

            /* Grid background (sayfa boyunca) */
            [data-testid="stAppViewContainer"]::before {
                content: '';
                position: fixed;
                inset: 0;
                pointer-events: none;
                z-index: 0;
                opacity: 0.15;
                background-image:
                    linear-gradient(var(--border) 1px, transparent 1px),
                    linear-gradient(90deg, var(--border) 1px, transparent 1px);
                background-size: 80px 80px;
                mask-image: radial-gradient(ellipse 80% 60% at 50% 50%, black, transparent);
            }

            /* Noise overlay */
            [data-testid="stAppViewContainer"]::after {
                content: '';
                position: fixed;
                inset: 0;
                pointer-events: none;
                z-index: 9999;
                opacity: 0.035;
                background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
            }

            [data-testid="stHeader"] { background: transparent; }

            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            #MainMenu,
            footer { display: none !important; visibility: hidden !important; }

            .block-container {
                position: relative;
                z-index: 1;
                max-width: 1200px;
                padding-top: 4.8rem;
                padding-bottom: 4rem;
            }

            /* ─── Top brand strip ─── */
            .top-brand {
                position: fixed;
                top: 0;
                left: 21rem;
                right: 0;
                z-index: 90;
                height: 58px;
                display: flex;
                align-items: center;
                padding: 0 clamp(20px, 4vw, 60px);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                background: rgba(10, 10, 11, 0.72);
                border-bottom: 1px solid var(--border);
            }

            .top-brand .top-logo,
            .top-brand .top-logo * {
                font-family: 'Instrument Serif', serif !important;
                font-size: 1.5rem !important;
                color: #ffffff !important;
                text-decoration: none !important;
                letter-spacing: -0.02em !important;
                line-height: 1 !important;
            }

            .top-brand .top-logo span {
                color: var(--accent) !important;
            }

            .top-brand .top-tag {
                margin-left: auto;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 6px 14px;
                border: 1px solid var(--border);
                border-radius: 60px;
                color: var(--accent);
                font-size: 0.68rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }

            .top-brand .top-tag::before {
                content: '';
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: var(--accent);
                box-shadow: 0 0 10px var(--accent);
            }

            @media (max-width: 768px) {
                .top-brand {
                    left: 18rem;
                }
            }

            /* ─── Sidebar tamamen gizli — kontroller ana sayfada ─── */
            section[data-testid="stSidebar"],
            [data-testid="collapsedControl"],
            [data-testid="stSidebarCollapsedControl"],
            [data-testid="stSidebarCollapseButton"] {
                display: none !important;
            }

            /* ─── Kontrol paneli (sidebar yerine) ─── */
            .controls-shell {
                position: relative;
                z-index: 5;
                background: linear-gradient(180deg, rgba(17,17,19,0.85), rgba(17,17,19,0.65));
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 18px 20px 10px;
                margin-bottom: 20px;
                backdrop-filter: blur(10px);
            }
            .controls-brand {
                font-family: 'Instrument Serif', serif;
                font-size: 1.6rem;
                color: #fff;
                letter-spacing: -0.02em;
                margin-bottom: 12px;
                line-height: 1;
            }
            .controls-brand span { color: var(--accent); }

            @media (max-width: 640px) {
                .controls-shell {
                    padding: 14px 14px 6px;
                    border-radius: 14px;
                }
                .controls-brand { font-size: 1.4rem; }
            }

            /* Eski sidebar stillerini placeholder olarak bırak (zarar yok) */
            section[data-testid="stSidebar"]-disabled {
                background: rgba(17, 17, 19, 0.96);
                border-right: 1px solid var(--border);
                min-width: 21rem !important;
                max-width: 21rem !important;
            }

            section[data-testid="stSidebar"] > div:first-child {
                width: 21rem !important;
                min-width: 21rem !important;
                max-width: 21rem !important;
            }

            section[data-testid="stSidebar"][aria-expanded="false"] {
                min-width: 21rem !important;
                max-width: 21rem !important;
                transform: none !important;
            }

            section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                margin-left: 0 !important;
            }

            section[data-testid="stSidebar"] button[kind="headerNoPadding"],
            section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] {
                display: none !important;
            }

            section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
                padding-top: 1.25rem;
            }

            /* ─── Cursor glow ─── */
            .cursor-glow {
                position: fixed;
                width: 300px;
                height: 300px;
                border-radius: 50%;
                pointer-events: none;
                z-index: 1;
                background: radial-gradient(circle, rgba(201, 240, 107, 0.06), transparent 70%);
                transform: translate(-50%, -50%);
                transition: left 0.15s ease, top 0.15s ease;
            }

            /* ─── Sidebar brand ─── */
            .sidebar-brand {
                padding: 20px 22px;
                border: 1px solid var(--border);
                border-radius: 20px;
                background: linear-gradient(180deg, rgba(26, 26, 30, 0.96), rgba(17, 17, 19, 0.92));
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.22);
                margin-bottom: 1.1rem;
            }

            .sidebar-brand .brand-mark,
            .sidebar-brand .brand-mark * {
                font-family: 'Instrument Serif', serif !important;
                font-size: 2.1rem !important;
                line-height: 1 !important;
                letter-spacing: -0.04em !important;
                color: #ffffff !important;
                text-decoration: none !important;
                border-bottom: none !important;
            }

            .sidebar-brand .brand-mark span {
                color: var(--accent) !important;
            }

            /* Google Translate override protection */
            .sidebar-brand .brand-mark font,
            .sidebar-brand .brand-mark font * {
                color: inherit !important;
                background: transparent !important;
            }

            .sidebar-control-label {
                margin-bottom: 0.7rem;
                color: var(--text-dim);
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.7rem;
                font-weight: 700;
            }

            /* ─── Typography ─── */
            h1, h2, h3 {
                color: var(--text) !important;
                font-family: 'Instrument Serif', serif !important;
                letter-spacing: -0.03em;
            }

            p, label, [data-testid="stMarkdownContainer"] { color: var(--text); }

            a { color: var(--accent); }

            /* ─── Language switcher (radio styled as flag pills) ─── */
            section[data-testid="stSidebar"] [data-testid="stRadio"] { margin-bottom: 0.25rem; }
            section[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 0.7rem; }

            section[data-testid="stSidebar"] [role="radiogroup"] {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.7rem;
            }

            section[data-testid="stSidebar"] [role="radiogroup"] label {
                margin: 0 !important;
                min-height: auto !important;
            }

            section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
                display: none !important;
            }

            section[data-testid="stSidebar"] [role="radiogroup"] label > div:last-child {
                width: 100%;
                position: relative;
                display: flex;
                align-items: center;
                justify-content: flex-start;
                min-height: 54px;
                padding: 0.9rem 1rem 0.9rem 3rem;
                border-radius: 14px;
                border: 1px solid var(--border);
                background: rgba(17, 17, 19, 0.94);
                color: var(--text);
                transition: border-color 0.25s ease, background 0.25s ease, transform 0.25s ease, box-shadow 0.25s ease;
            }

            section[data-testid="stSidebar"] [role="radiogroup"] label > div:last-child p {
                margin: 0 !important;
                color: var(--text) !important;
                font-size: 0.92rem !important;
                font-weight: 700 !important;
                line-height: 1.1 !important;
            }

            section[data-testid="stSidebar"] [role="radiogroup"] label > div:last-child::before {
                content: '';
                position: absolute;
                left: 14px;
                top: 50%;
                width: 26px;
                height: 18px;
                transform: translateY(-50%);
                border-radius: 5px;
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.28);
            }

            section[data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(1) > div:last-child::before {
                background-image: url("__FLAG_TR__");
            }

            section[data-testid="stSidebar"] [role="radiogroup"] label:nth-of-type(2) > div:last-child::before {
                background-image: url("__FLAG_EN__");
            }

            section[data-testid="stSidebar"] [role="radiogroup"] label:hover > div:last-child {
                border-color: rgba(201, 240, 107, 0.4);
                transform: translateY(-1px);
            }

            section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) > div:last-child {
                border-color: rgba(201, 240, 107, 0.7);
                background: linear-gradient(180deg, rgba(201, 240, 107, 0.98), rgba(179, 224, 79, 0.98));
                box-shadow: 0 12px 28px rgba(201, 240, 107, 0.22);
            }

            section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) > div:last-child p {
                color: #0b0b0d !important;
            }

            /* ─── Buttons ─── */
            .stButton > button {
                border: none !important;
                border-radius: 60px !important;
                background: var(--accent) !important;
                color: #0b0b0d !important;
                font-weight: 700 !important;
                letter-spacing: 0.02em !important;
                padding: 0.82rem 1.3rem !important;
                transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.25s ease !important;
                box-shadow: 0 10px 30px rgba(201, 240, 107, 0.14);
            }

            .stButton > button:hover {
                transform: translateY(-2px);
                background: #d4f57a !important;
                box-shadow: 0 16px 34px rgba(201, 240, 107, 0.24);
            }

            .stButton > button *,
            .stFormSubmitButton > button *,
            .stFileUploader button * {
                color: #0b0b0d !important;
                fill: #0b0b0d !important;
                stroke: #0b0b0d !important;
                font-weight: 700 !important;
            }

            .stFormSubmitButton > button,
            .stFileUploader button {
                border-radius: 60px !important;
                background: var(--accent) !important;
                color: #0b0b0d !important;
                border: none !important;
                box-shadow: 0 10px 28px rgba(201, 240, 107, 0.14);
            }

            .stFormSubmitButton > button:hover,
            .stFileUploader button:hover {
                background: #d4f57a !important;
                color: #0b0b0d !important;
            }

            .stButton > button svg,
            .stFormSubmitButton > button svg,
            .stFileUploader button svg {
                color: #0b0b0d !important;
                fill: #0b0b0d !important;
                stroke: #0b0b0d !important;
            }

            /* ─── Inputs ─── */
            .stTextInput input,
            .stTextArea textarea,
            [data-baseweb="input"] input,
            [data-baseweb="textarea"] textarea,
            [data-baseweb="base-input"],
            [data-baseweb="select"] > div,
            .stFileUploader > div {
                background: rgba(17, 17, 19, 0.92) !important;
                color: var(--text) !important;
                border: 1px solid var(--border) !important;
                border-radius: 14px !important;
            }

            .stTextInput input,
            .stTextArea textarea,
            [data-baseweb="input"] input,
            [data-baseweb="textarea"] textarea {
                color: #f3efe8 !important;
                -webkit-text-fill-color: #f3efe8 !important;
                caret-color: var(--accent) !important;
                opacity: 1 !important;
            }

            .stTextInput input::placeholder,
            .stTextArea textarea::placeholder,
            [data-baseweb="input"] input::placeholder,
            [data-baseweb="textarea"] textarea::placeholder {
                color: #8e8a83 !important;
                -webkit-text-fill-color: #8e8a83 !important;
                opacity: 1 !important;
            }

            .stTextInput input:focus,
            .stTextArea textarea:focus,
            [data-baseweb="input"] input:focus,
            [data-baseweb="textarea"] textarea:focus,
            [data-baseweb="base-input"]:focus-within {
                border-color: var(--accent) !important;
                box-shadow: 0 0 0 0.18rem rgba(201, 240, 107, 0.12) !important;
            }

            /* ─── File uploader (dark + dashed) ─── */
            [data-testid="stFileUploader"],
            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploaderDropzone"],
            [data-testid="stFileUploaderDropzoneInstructions"],
            .stFileUploader,
            .stFileUploader > div,
            .stFileUploader > div > div,
            .stFileUploader section {
                background: linear-gradient(180deg, rgba(17, 17, 19, 0.96), rgba(10, 10, 11, 0.96)) !important;
                background-color: rgba(17, 17, 19, 0.96) !important;
                color: var(--text) !important;
                border-radius: 14px !important;
            }

            [data-testid="stFileUploaderDropzone"],
            .stFileUploader section {
                border: 1px dashed var(--border) !important;
                padding: 14px !important;
            }

            [data-testid="stFileUploader"] small,
            [data-testid="stFileUploader"] span,
            [data-testid="stFileUploader"] p,
            [data-testid="stFileUploaderDropzoneInstructions"] * {
                color: var(--text-dim) !important;
            }

            /* Yüklenen dosya kartı */
            [data-testid="stFileUploaderFile"],
            [data-testid="stFileUploaderFile"] > div,
            [data-testid="stFileUploaderFileName"] {
                background: rgba(26, 26, 30, 0.96) !important;
                color: var(--text) !important;
                border-radius: 12px !important;
            }

            [data-testid="stFileUploaderFileName"],
            [data-testid="stFileUploaderFile"] * {
                color: var(--text) !important;
            }

            /* ─── Tabs ─── */
            .stTabs [data-baseweb="tab-list"] {
                background: rgba(17, 17, 19, 0.92);
                border: 1px solid var(--border);
                border-radius: 60px;
                padding: 6px;
                gap: 0.4rem;
                width: fit-content;
                max-width: 100%;
            }

            .stTabs [data-baseweb="tab"] {
                min-height: 44px;
                padding: 0 22px;
                border-radius: 60px;
                border: 1px solid transparent;
                background: transparent;
                color: var(--text-dim);
                font-weight: 600;
                font-family: 'DM Sans', sans-serif;
                transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
            }

            .stTabs [aria-selected="true"] {
                background: var(--accent) !important;
                border-color: var(--accent) !important;
                box-shadow: 0 8px 22px rgba(201, 240, 107, 0.18);
                color: #0b0b0d !important;
            }

            .stTabs [data-baseweb="tab"] p {
                color: #c8c4bc !important;
                font-size: 0.92rem !important;
                font-weight: 600 !important;
                line-height: 1 !important;
                margin: 0 !important;
            }

            .stTabs [aria-selected="true"] p { color: #0b0b0d !important; }

            .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

            .stTabs [data-baseweb="tab"]:hover {
                border-color: rgba(201, 240, 107, 0.24);
                color: var(--text);
            }

            /* ─── Metrics ─── */
            [data-testid="stMetric"] {
                background: var(--bg2);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 18px 22px;
                transition: border-color 0.3s, transform 0.3s;
            }

            [data-testid="stMetric"]:hover {
                border-color: var(--accent);
                transform: translateY(-2px);
            }

            [data-testid="stMetricLabel"] {
                color: var(--text-dim) !important;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                font-size: 0.72rem !important;
            }

            [data-testid="stMetricValue"] {
                color: var(--accent) !important;
                font-family: 'Instrument Serif', serif !important;
                letter-spacing: -0.02em;
                font-size: 2.2rem !important;
            }

            .stProgress > div > div > div {
                background: linear-gradient(90deg, var(--accent), #d4f57a) !important;
            }

            hr { border-color: var(--border) !important; }

            /* ─── Hero ─── */
            .hero-shell {
                position: relative;
                overflow: hidden;
                padding: clamp(2.2rem, 5vw, 3.6rem);
                border-radius: 24px;
                border: 1px solid var(--border);
                background: linear-gradient(180deg, rgba(17, 17, 19, 0.6), rgba(10, 10, 11, 0.6));
                margin-bottom: 1.5rem;
            }

            /* Floating orbs */
            .orb {
                position: absolute;
                border-radius: 50%;
                filter: blur(80px);
                opacity: 0.18;
                pointer-events: none;
                animation: float 8s ease-in-out infinite;
            }

            .orb-1 {
                width: 360px; height: 360px;
                background: var(--accent);
                top: -60px; right: -60px;
                animation-delay: 0s;
            }

            .orb-2 {
                width: 260px; height: 260px;
                background: var(--accent2);
                bottom: -80px; left: -40px;
                animation-delay: 3s;
            }

            .orb-3 {
                width: 180px; height: 180px;
                background: var(--accent3);
                top: 40%; right: 30%;
                animation-delay: 5s;
            }

            .hero-label {
                position: relative;
                z-index: 2;
                display: inline-flex;
                align-items: center;
                gap: 10px;
                color: var(--accent);
                text-transform: uppercase;
                letter-spacing: 0.16em;
                font-size: 0.76rem;
                font-weight: 700;
                margin-bottom: 14px;
            }

            .hero-label::before {
                content: '';
                width: 22px;
                height: 2px;
                background: var(--accent);
            }

            .hero-title {
                position: relative;
                z-index: 2;
                margin: 0;
                font-family: 'Instrument Serif', serif;
                font-size: clamp(3rem, 8vw, 5.6rem);
                line-height: 0.95;
                letter-spacing: -0.05em;
            }

            .hero-title em {
                color: var(--accent);
                font-style: italic;
            }

            .hero-meta {
                position: relative;
                z-index: 2;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 1.8rem;
            }

            .meta-pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 9px 16px;
                border-radius: 999px;
                border: 1px solid var(--border);
                background: rgba(26, 26, 30, 0.78);
                color: var(--text-dim);
                font-size: 0.74rem;
                font-weight: 600;
                letter-spacing: 0.1em;
                text-transform: uppercase;
            }

            .meta-pill strong {
                color: var(--text);
                font-weight: 700;
            }

            /* ─── Marquee ─── */
            .marquee-wrap {
                overflow: hidden;
                margin: 1rem 0 2rem;
                padding: 16px 0;
                border-top: 1px solid var(--border);
                border-bottom: 1px solid var(--border);
            }

            .marquee {
                display: flex;
                gap: 28px;
                width: max-content;
                animation: marquee 25s linear infinite;
            }

            .marquee span {
                display: inline-flex;
                align-items: center;
                color: var(--text-dim);
                font-family: 'Instrument Serif', serif;
                font-size: 1.1rem;
                white-space: nowrap;
            }

            .marquee span::after {
                content: '◆';
                margin-left: 28px;
                font-size: 0.48rem;
                color: var(--accent);
            }

            @keyframes marquee {
                from { transform: translateX(0); }
                to { transform: translateX(-50%); }
            }

            @keyframes float {
                0%, 100% { transform: translate(0, 0); }
                50% { transform: translate(20px, -30px); }
            }

            /* ─── Empty state ─── */
            .empty-shell {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: clamp(2.2rem, 6vw, 4rem) 2rem;
                border: 1px dashed var(--border);
                border-radius: 24px;
                background: rgba(17, 17, 19, 0.5);
                text-align: center;
            }

            .empty-shell .icon {
                font-size: 2.4rem;
                margin-bottom: 0.8rem;
            }

            .empty-shell h2 {
                font-family: 'Instrument Serif', serif;
                font-size: clamp(1.6rem, 3.2vw, 2.2rem);
                margin: 0;
            }

            .empty-shell h2 em {
                color: var(--accent);
                font-style: italic;
            }

            /* ─── Chat card ─── */
            .chat-card {
                margin-top: 1rem;
                padding: 20px 22px;
                border-radius: 16px;
                border: 1px solid var(--border);
                background: rgba(17, 17, 19, 0.9);
                transition: border-color 0.3s;
            }

            .chat-card:hover { border-color: var(--accent); }

            .chat-card-label {
                color: var(--accent);
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.72rem;
                font-weight: 700;
            }

            .chat-card-question {
                margin-top: 0.55rem;
                font-size: 1rem;
                line-height: 1.7;
                color: var(--text);
            }

            .chat-answer-spacer {
                height: 0.65rem;
            }

            code { font-family: 'JetBrains Mono', monospace; }

            /* ─── Scrollbar ─── */
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: var(--bg); }
            ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

            /* ─── Responsive: tablet ve daha küçük (≤768px) ─── */
            @media (max-width: 768px) {
                .block-container {
                    padding-top: 4.8rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .hero-shell {
                    padding: 1.6rem;
                    border-radius: 20px;
                }

                .hero-title { font-size: 2.4rem; }
                .orb-1 { width: 220px; height: 220px; }
                .orb-2 { width: 160px; height: 160px; }
                .orb-3 { width: 120px; height: 120px; }

                /* Üst marka çubuğu: sidebar açıkken bile pencereye sığsın */
                .top-brand {
                    left: 0 !important;
                    height: 52px;
                    padding: 0 16px;
                }
                .top-brand .top-logo,
                .top-brand .top-logo * { font-size: 1.25rem !important; }
                .top-brand .top-tag { padding: 4px 10px; font-size: 11px; }
            }

            /* ─── Responsive: telefon (≤640px) ─── */
            @media (max-width: 640px) {
                /* Top-brand mobilde gizli — hamburger butonu için yer açıyor.
                   Marka zaten ana hero'da büyük şekilde görünüyor. */
                .top-brand { display: none !important; }
                .block-container { padding-top: 4rem !important; }
                /* NOT: sidebar mobil davranışı force_open_sidebar JS'inde
                   tanımlı — özel hamburger butonu + transform ile yönetiliyor. */

                .block-container {
                    padding-top: 4rem;
                    padding-left: 0.75rem;
                    padding-right: 0.75rem;
                    max-width: 100%;
                }

                .hero-shell { padding: 1.1rem; border-radius: 16px; }
                .hero-title { font-size: 1.9rem; line-height: 1.1; }
                .hero-label { font-size: 11px; }

                .orb-1 { width: 140px; height: 140px; opacity: 0.5; }
                .orb-2 { width: 110px; height: 110px; opacity: 0.4; }
                .orb-3 { width: 80px; height: 80px; opacity: 0.3; }

                /* Top brand: tam akıllı telefon ekranına sığsın */
                .top-brand { height: 48px; padding: 0 12px; }
                .top-brand .top-logo,
                .top-brand .top-logo * { font-size: 1.1rem !important; }
                .top-brand .top-tag {
                    padding: 3px 8px;
                    font-size: 10px;
                    letter-spacing: 0.04em;
                }

                /* Sayfa ve sohbet kartlarındaki bilgi karoları */
                .stat-card, .pdf-meta {
                    font-size: 12px !important;
                    padding: 6px 10px !important;
                }

                /* Tabs (Özet / Sohbet) — küçük ekranda daha sıkışık */
                .stTabs [data-baseweb="tab-list"] {
                    gap: 4px;
                    padding: 4px;
                }
                .stTabs [data-baseweb="tab"] {
                    padding: 8px 14px !important;
                    font-size: 14px !important;
                }

                /* Bütün butonlar full-width hissedilsin */
                .stButton button, .stDownloadButton button {
                    width: 100% !important;
                    font-size: 14px !important;
                    padding: 12px 16px !important;
                }

                /* Marquee gizle — yer kaplıyor */
                .marquee { display: none !important; }

                /* PDF meta bilgi karoları stack */
                .stat-row, .pdf-info-row {
                    flex-direction: column !important;
                    gap: 8px !important;
                    align-items: stretch !important;
                }

                /* Chat kartı içi padding'i azalt */
                .chat-card {
                    padding: 12px 14px !important;
                    border-radius: 12px !important;
                }
                .chat-card-question { font-size: 14px !important; }
            }

            /* ─── TTS widget: her ekranda akıllı sarılma ─── */
            @media (max-width: 520px) {
                /* TTS widget iframe içinde — bu CSS oraya ulaşmaz ama yine de
                   ana sayfada üzerine yapışan yüksekliği biraz büyütelim ki
                   iframe sıkışmasın */
                iframe[title="streamlit_app"] { min-height: 110px; }
            }
        </style>
        """
    css = css.replace("__FLAG_TR__", FLAG_DATA_URIS["tr"]).replace("__FLAG_EN__", FLAG_DATA_URIS["en"])
    st.markdown(css, unsafe_allow_html=True)


def render_cursor_glow() -> None:
    st.markdown('<div class="cursor-glow" id="cursor-glow"></div>', unsafe_allow_html=True)


def render_top_brand() -> None:
    st.markdown(
        '<div class="top-brand" translate="no">'
        '<a href="#" class="top-logo" translate="no">mtk<span>.</span></a>'
        '<div class="top-tag" translate="no">PDF Assistant</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def inject_cursor_script() -> None:
    # Cursor glow takip scripti — parent belgeye uygulanır
    components.html(
        """
        <script>
            (function() {
                const parentDoc = window.parent.document;
                const glow = parentDoc.getElementById('cursor-glow');
                if (!glow) return;
                if (glow.dataset.bound === '1') return;
                glow.dataset.bound = '1';
                parentDoc.addEventListener('mousemove', (e) => {
                    glow.style.left = e.clientX + 'px';
                    glow.style.top = e.clientY + 'px';
                });
            })();
        </script>
        """,
        height=0,
    )


def render_sidebar_brand() -> None:
    st.markdown(
        '<div class="sidebar-brand" translate="no">'
        '<div class="brand-mark" translate="no">mtk<span>.</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_language_switcher() -> None:
    st.markdown(
        f'<div class="sidebar-control-label">{escape(t("language_label"))}</div>',
        unsafe_allow_html=True,
    )
    st.radio(
        t("language_label"),
        options=["tr", "en"],
        format_func=get_language_name,
        horizontal=True,
        label_visibility="collapsed",
        key="lang_selector",
        on_change=handle_language_change,
    )


def force_open_sidebar() -> None:
    """No-op: sidebar artık kullanılmıyor (kontroller ana sayfada).
    Streamlit sidebar'ı CSS'le tamamen gizleniyor."""
    pass

def render_hero() -> None:
    has_pdf = bool(st.session_state.chunks)
    page_count = len(st.session_state.chunks or [])
    file_name = st.session_state.last_file_name or "—"

    hero_html = (
        '<section class="hero-shell">'
        '<div class="orb orb-1"></div>'
        '<div class="orb orb-2"></div>'
        '<div class="orb orb-3"></div>'
        f'<div class="hero-label" translate="no">{escape(t("hero_label"))}</div>'
        f'<h1 class="hero-title">{t("hero_title")}</h1>'
        '<div class="hero-meta">'
        f'<div class="meta-pill">{escape(t("pages"))}: <strong>{page_count if has_pdf else "0"}</strong></div>'
        f'<div class="meta-pill">PDF: <strong>{escape(file_name)}</strong></div>'
        "</div>"
        "</section>"
    )
    st.markdown(hero_html, unsafe_allow_html=True)


def render_marquee() -> None:
    items = ["C++", "C#", "Java", "Python", "AI", "PDF", "API", "Llama"]
    repeated = items + items + items
    content = "".join(f'<span translate="no">{escape(item)}</span>' for item in repeated)

    st.markdown(
        f'<div class="marquee-wrap"><div class="marquee">{content}</div></div>',
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        '<div class="empty-shell">'
        '<div>'
        '<div class="icon">📄</div>'
        f'<h2>{t("hero_title")}</h2>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def estimate_summary_seconds(chunks: List[Dict]) -> int:
    """Kullanıcıya gösterilecek, PDF büyüklüğüne dayalı tahmini süre (sn).

    Heuristic — Groq llama-3.1-8b-instant pratik ölçümlerinden:
      • Prompt ingest: ~3000 token/s (1 char ≈ 0.25 token → 12000 char/sn)
      • Output gen:    ~700 token/s
      • Network + queue overhead: ~3-4 sn

    Eksik tahmin "sayaç bitti ama özet hala gelmiyor" hissi yaratır —
    bu yüzden tahminler bilinçli olarak gerçek beklenenden yüksek tutulur.
    Streamlit'te ilk token gelene kadar timer görsel olarak donuktur, o
    yüzden tahmin kullanıcının sabırlı kalmasına yetecek kadar uzun olmalı.
    """
    total_chars = sum(len(c["text"]) for c in chunks)
    if total_chars <= DIRECT_SUMMARY_CHAR_LIMIT:
        # Direct: ingest_sec + output_sec + overhead
        # 3B-preview ile: ingest ~5000 char/sn (4x daha hızlı), output ~1500 tok/sn
        # 8B fallback olursa süre uzar — tahmin best-case yapılıyor, overrun
        # durumunda "+X sn" gösterimi kullanıcıyı haberdar eder.
        ingest_sec = total_chars // 18000          # 50k char → ~3 sn
        output_sec = 3                              # 2500 tok @ 1500/s ≈ 1.7 sn → yuvarlak 3
        overhead = 4                                # network + queue + ilk token gecikmesi
        return max(8, min(30, ingest_sec + output_sec + overhead))

    # Chunked: paralel batch ingest + final birleştirme
    num_pieces = max(1, total_chars // CHUNK_CHAR_LIMIT + 1)
    parallel_batches = (num_pieces + MAX_PARALLEL_CHUNK_REQUESTS - 1) // MAX_PARALLEL_CHUNK_REQUESTS
    # Her batch ~5 sn (ingest+output), final ~8 sn, baseline 6 sn
    return max(15, parallel_batches * 5 + 8)


def render_summary_tab() -> None:
    st.session_state.setdefault("summary_cache", {})
    st.session_state.setdefault("summary_pdf_cache", {})
    if st.button(t("btn_summary"), use_container_width=True):
        cache_key = summary_cache_key()
        cached_summary = st.session_state.summary_cache.get(cache_key)
        if cached_summary:
            st.session_state.summary = cached_summary
            st.rerun()
            return

        # Akış halinde özet — kullanıcı ilk token'ları saniyeler içinde görür.
        placeholder = st.empty()
        estimated = estimate_summary_seconds(st.session_state.chunks)

        # Fallback zinciri — pratikte denenmiş güvenilir sıra:
        #   8B-instant (≈700 tok/s) → 70B-versatile (≈250 tok/s)
        # FAST_SUMMARY_MODEL None değilse zincirin başına eklenir.
        # Bu session'da başarısız olmuş model'ler atlanır → ikinci tıklamada
        # bilinen kötü model tekrar denenmez.
        st.session_state.setdefault("failed_summary_models", set())
        failed: set = st.session_state.failed_summary_models

        seen: set = set()
        models_to_try: List[str] = []
        for candidate in (FAST_SUMMARY_MODEL, SUMMARY_MODEL_NAME, CHAT_MODEL_NAME):
            if candidate and candidate not in seen and candidate not in failed:
                models_to_try.append(candidate)
                seen.add(candidate)

        # Tüm model'ler önceki denemelerde başarısız olduysa, en kararlısını
        # (70B) yine de bir kez daha dene — geçici hatalar olabilir.
        if not models_to_try:
            models_to_try = [CHAT_MODEL_NAME]
            failed.clear()  # bir sonraki sefer için temizle

        result = ""
        for index, model in enumerate(models_to_try):
            is_last = index == len(models_to_try) - 1
            generator = stream_summarize_pdf(st.session_state.chunks, model)
            result = _stream_to_placeholder(
                generator,
                placeholder,
                show_errors=is_last,
                estimated_sec=estimated,
            )
            if result:
                break
            # Bu modeli başarısız olarak işaretle — aynı session'da tekrar denenmesin.
            failed.add(model)
            if not is_last:
                # Sessizce yedek modele geç, kullanıcıya kısa bilgi ver.
                st.info(t("fallback_notice"))
                placeholder = st.empty()

        if not result:
            st.session_state.summary = None
            return

        st.session_state.summary = result
        st.session_state.summary_cache[cache_key] = result
        # rerun → final görünüm (PDF indir butonu + render edilmiş özet) çıksın.
        st.rerun()
        return

    if st.session_state.summary:
        pdf_cache_key = summary_pdf_cache_key(st.session_state.summary)
        pdf_bytes = st.session_state.summary_pdf_cache.get(pdf_cache_key)
        if not pdf_bytes:
            pdf_bytes = create_summary_pdf_bytes(
                st.session_state.summary,
                st.session_state.last_file_name,
            )
            st.session_state.summary_pdf_cache[pdf_cache_key] = pdf_bytes
        st.download_button(
            t("btn_download_summary"),
            data=pdf_bytes,
            file_name=_summary_pdf_filename(st.session_state.last_file_name),
            mime="application/pdf",
            use_container_width=True,
        )
        render_tts_widget(st.session_state.summary)
        st.markdown(st.session_state.summary)


def _show_placeholder_error(placeholder, message: str) -> None:
    if hasattr(placeholder, "error"):
        placeholder.error(message)
    else:
        placeholder.markdown(message)


def _format_timer_label(start_ts: float, estimated_sec: Optional[int]) -> str:
    """Stream sırasında üst kısma yazılacak geri sayım/elapsed etiketi."""
    elapsed = max(0, int(time.monotonic() - start_ts))
    if estimated_sec is None:
        return f"⏱ {elapsed} sn"
    remaining = estimated_sec - elapsed
    if remaining > 0:
        return f"⏱ {t('timer_remaining').format(secs=remaining)}"
    return f"⏱ {t('timer_overrun').format(secs=-remaining)}"


def _stream_to_placeholder(
    generator,
    placeholder,
    show_errors: bool = True,
    estimated_sec: Optional[int] = None,
) -> str:
    """stream_llm çıktısını incrementally render eder; tam metni döner.

    `estimated_sec` verilirse stream sırasında üstte geri sayım/elapsed gösterilir.
    İlk token gelmeden de sayaç güncellensin diye arka planda yarım saniyede bir
    placeholder'ı yeniden çizen bir ticker thread çalışır (best-effort; streamlit
    context attach edilemezse sessizce devre dışı kalır).

    Hata olduğunda gerçek API mesajı `expander` içinde kullanıcıya açılır
    (jenerik mesajın altında) — teşhis edilebilirlik için kritik.
    """
    import threading

    buffer = ""
    error_text: Optional[str] = None
    start_ts = time.monotonic()

    # Mutable state, ticker thread'den okunup yazılır (GIL koruması yeterli).
    state = {"buffer": "", "done": False}

    def _render_now() -> None:
        b = state["buffer"]
        if estimated_sec is not None:
            label = _format_timer_label(start_ts, estimated_sec)
            if b:
                placeholder.markdown(f"{label}\n\n{b}▌")
            else:
                placeholder.markdown(f"{label} · _{t('processing')}_")
        else:
            if b:
                placeholder.markdown(b + "▌")
            else:
                placeholder.markdown(t("processing"))

    # İlk çizim — kullanıcı butona basar basmaz timer'ı görsün.
    _render_now()

    # Arka plan ticker — best-effort, hata olursa sessizce sön.
    ticker_thread: Optional[threading.Thread] = None
    if estimated_sec is not None:
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx  # type: ignore

            def _tick() -> None:
                while not state["done"]:
                    time.sleep(0.5)
                    if state["done"]:
                        return
                    try:
                        _render_now()
                    except Exception:
                        return

            ticker_thread = threading.Thread(target=_tick, daemon=True)
            try:
                add_script_run_ctx(ticker_thread)
            except Exception:
                pass
            ticker_thread.start()
        except Exception:
            ticker_thread = None

    try:
        for piece in generator:
            if not piece:
                continue
            if piece.startswith(LLM_ERROR_PREFIX):
                error_text = piece[len(LLM_ERROR_PREFIX):]
                break
            buffer += piece
            state["buffer"] = buffer
            _render_now()
    finally:
        state["done"] = True
        if ticker_thread is not None:
            ticker_thread.join(timeout=0.2)

    if not buffer:
        message = t("llm_failed") if error_text is not None else t("llm_empty")
        if show_errors:
            _show_placeholder_error(placeholder, message)
            # Geliştirici için: gerçek API hatasını expander içinde göster.
            if error_text:
                try:
                    with st.expander(t("error_detail"), expanded=False):
                        st.code(error_text)
                except Exception:
                    # Test stub'unda expander olmayabilir; sessizce geç.
                    pass
        else:
            placeholder.empty()
        return ""

    placeholder.markdown(buffer)
    if error_text is not None:
        # Akış sırasında kesildi ama bir şey aldık — yine de uyarı ver
        st.warning(t("llm_failed"))
    return buffer


def render_chat_tab() -> None:
    st.session_state.setdefault("chat_cache", {})
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            t("placeholder_chat"),
            placeholder=t("placeholder_chat"),
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(t("btn_ask"), use_container_width=True)

    if submitted:
        if question.strip():
            q = question.strip()
            st.markdown(
                f'<div class="chat-card">'
                f'<div class="chat-card-label">{escape(t("question_label"))}</div>'
                f'<div class="chat-card-question">{escape(q)}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            cache_key = chat_cache_key(q, st.session_state.chat_history)
            cached_answer = st.session_state.chat_cache.get(cache_key)
            if cached_answer:
                st.session_state.chat_history.append({"q": q, "a": cached_answer})
                st.rerun()
                return
            st.markdown('<div class="chat-answer-spacer"></div>', unsafe_allow_html=True)
            placeholder = st.empty()
            generator = chat_with_pdf_stream(
                q, st.session_state.chunks, MODEL_NAME,
                history=st.session_state.chat_history,
            )
            answer = _stream_to_placeholder(generator, placeholder)
            if answer:
                st.session_state.chat_cache[cache_key] = answer
                st.session_state.chat_history.append({"q": q, "a": answer})
                st.rerun()
                return
        else:
            st.warning(t("no_question"))

    if st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history):
            st.markdown(
                f'<div class="chat-card">'
                f'<div class="chat-card-label">{escape(t("question_label"))}</div>'
                f'<div class="chat-card-question">{escape(chat["q"])}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chat-answer-spacer"></div>', unsafe_allow_html=True)
            st.markdown(chat["a"])
            st.markdown("---")


# Main app
ensure_state()
inject_theme()
render_top_brand()
render_cursor_glow()
inject_cursor_script()

# ─── Kontrol paneli — sidebar yerine ana akışın başında ──────────────
# Dil seçici + PDF yükleme + Temizle. Hem desktop hem mobilde aynı yerde.
# Streamlit sidebar'ını CSS ile gizliyoruz (hamburger karmaşası yok).
st.markdown('<div class="controls-shell">', unsafe_allow_html=True)
st.markdown(f'<div class="controls-brand">mtk<span>.</span></div>', unsafe_allow_html=True)

ctrl_lang, ctrl_upload, ctrl_clear = st.columns([1, 2, 1])

with ctrl_lang:
    render_language_switcher()

with ctrl_upload:
    uploaded_file = st.file_uploader(
        t("upload"),
        type=["pdf"],
        label_visibility="visible",
    )

with ctrl_clear:
    if st.button(t("clear"), use_container_width=True, key="btn_clear_main"):
        clear_workspace()
        st.rerun()

if uploaded_file is None and st.session_state.last_file_signature is not None:
    clear_workspace()

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_signature = hashlib.sha1(file_bytes).hexdigest()
    if st.session_state.last_file_signature != file_signature:
        try:
            new_chunks = extract_pdf_chunks(file_bytes)
        except PDFLoadError as err:
            st.error(t(err.key).format(**err.fmt))
            clear_workspace()
        else:
            st.session_state.chunks = new_chunks
            st.session_state.summary = None
            st.session_state.chat_history = []
            st.session_state.last_file_signature = file_signature
            st.session_state.last_file_name = uploaded_file.name

            total_chars = sum(len(c["text"]) for c in new_chunks)
            if total_chars > MAX_PDF_TEXT_CHARS:
                st.warning(t("warn_huge_text").format(chars=f"{total_chars:,}"))

if not GROQ_API_KEY:
    st.error(t("no_api_key"))

st.markdown('</div>', unsafe_allow_html=True)

render_hero()
render_marquee()

if st.session_state.chunks:
    summary_tab, chat_tab = st.tabs([t("summary_tab"), t("chat_tab")])

    with summary_tab:
        render_summary_tab()

    with chat_tab:
        render_chat_tab()
else:
    render_empty_state()