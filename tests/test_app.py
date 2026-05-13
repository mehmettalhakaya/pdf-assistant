"""Unit tests for the PDF Assistant app.

These tests exercise pure logic (retry math, RAG scoring, chunking, prompt
building) plus the LLM helpers with a mocked Groq client. They never reach the
network and never spin up Streamlit.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for building fake Groq responses / streams
# ──────────────────────────────────────────────────────────────────────────────

def _resp(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _stream(deltas):
    for d in deltas:
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=d))]
        )


def _install_client(app, side_effect=None, return_value=None):
    """Replace app.client with a MagicMock returning the given values."""
    fake = MagicMock()
    if side_effect is not None:
        fake.chat.completions.create.side_effect = side_effect
    elif return_value is not None:
        fake.chat.completions.create.return_value = return_value
    app.client = fake
    return fake


class _Placeholder:
    def __init__(self):
        self.markdowns = []
        self.errors = []
        self.empty_calls = 0
        self.last = None

    def markdown(self, body=None, *_args, **_kwargs):
        self.markdowns.append(body)
        self.last = body

    def error(self, body=None, *_args, **_kwargs):
        self.errors.append(body)
        self.last = body

    def empty(self):
        self.empty_calls += 1
        self.last = None


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make every retry test instantaneous."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)


# ──────────────────────────────────────────────────────────────────────────────
# _extract_wait_seconds
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractWaitSeconds:
    def test_seconds_format(self, app_module):
        assert app_module._extract_wait_seconds("try again in 3s") == pytest.approx(3.5)

    def test_milliseconds_format(self, app_module):
        v = app_module._extract_wait_seconds("retry in 500ms please")
        assert v == pytest.approx(0.7, abs=0.01)

    def test_clamped_to_max_single(self, app_module):
        v = app_module._extract_wait_seconds("try again in 9999s")
        assert v == app_module.MAX_SINGLE_WAIT_SEC

    def test_default_when_unparseable(self, app_module):
        assert app_module._extract_wait_seconds("nope") == 12.0

    def test_case_insensitive(self, app_module):
        assert app_module._extract_wait_seconds("TRY AGAIN IN 2S") == pytest.approx(2.5)


# ──────────────────────────────────────────────────────────────────────────────
# _is_retryable_error
# ──────────────────────────────────────────────────────────────────────────────

class TestIsRetryable:
    @pytest.mark.parametrize("msg", [
        "rate_limit_exceeded",
        "Rate limit reached",
        "Status 429: too many",
        "413 Payload Too Large",
        "tokens per minute exceeded",
        "request timeout",
        "Connection reset",
        "503 service unavailable",
        "model is overloaded",
        "internal server error",
        "504 gateway timeout",
    ])
    def test_retryable(self, app_module, msg):
        assert app_module._is_retryable_error(msg) is True

    @pytest.mark.parametrize("msg", [
        "invalid api key",
        "model not found",
        "400 bad request: malformed prompt",
        "permission denied",
        "",
    ])
    def test_non_retryable(self, app_module, msg):
        assert app_module._is_retryable_error(msg) is False


# ──────────────────────────────────────────────────────────────────────────────
# _score_tokens
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreTokens:
    def test_lowercases_and_filters_short(self, app_module):
        toks = app_module._score_tokens("Bu BIR Test")
        # "bu" filtered (stopword), "bir" filtered (stopword), "test" kept
        assert "test" in toks
        assert "bu" not in toks
        assert "bir" not in toks

    def test_filters_2char_tokens(self, app_module):
        toks = app_module._score_tokens("ai is on")
        # All <3 chars → empty
        assert toks == set()

    def test_unicode_words(self, app_module):
        toks = app_module._score_tokens("Algoritma ve Programlama")
        assert "algoritma" in toks
        assert "programlama" in toks
        assert "ve" not in toks  # too short anyway


# ──────────────────────────────────────────────────────────────────────────────
# build_context
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildContext:
    def test_empty_chunks(self, app_module):
        assert app_module.build_context([]) == ""
        assert app_module.build_context([], question="x") == ""

    def test_no_question_takes_from_start(self, app_module):
        chunks = [
            {"page": 1, "text": "alpha"},
            {"page": 2, "text": "beta"},
            {"page": 3, "text": "gamma"},
        ]
        ctx = app_module.build_context(chunks, question="", max_chars=10_000)
        assert ctx.index("[Page 1]") < ctx.index("[Page 2]") < ctx.index("[Page 3]")

    def test_question_prefers_relevant_chunk(self, app_module):
        chunks = [
            {"page": 1, "text": "kediler ve kopekler hakkinda"},
            {"page": 2, "text": "elma armut portakal"},
            {"page": 3, "text": "araba motor benzin"},
        ]
        # Tight budget so only the most relevant survives
        ctx = app_module.build_context(chunks, question="elma fiyati", max_chars=40)
        assert "[Page 2]" in ctx
        assert "[Page 1]" not in ctx
        assert "[Page 3]" not in ctx

    def test_picked_chunks_sorted_by_page(self, app_module):
        # Both relevant — but page 5 has more overlap; result must still be page-ordered
        chunks = [
            {"page": 5, "text": "elma elma elma"},
            {"page": 2, "text": "elma kiraz"},
            {"page": 9, "text": "alakasiz konu"},
        ]
        ctx = app_module.build_context(chunks, question="elma", max_chars=10_000)
        assert ctx.index("[Page 2]") < ctx.index("[Page 5]") < ctx.index("[Page 9]")

    def test_no_overlap_falls_back_to_first(self, app_module):
        # Even with zero overlap, we still must return *some* context
        chunks = [{"page": 1, "text": "kedi"}, {"page": 2, "text": "kopek"}]
        ctx = app_module.build_context(chunks, question="elma", max_chars=10_000)
        assert "[Page 1]" in ctx and "[Page 2]" in ctx

    def test_small_context_skips_token_scoring(self, app_module, monkeypatch):
        def fail_score(_text):
            raise AssertionError("token scoring should not run for tiny PDFs")

        monkeypatch.setattr(app_module, "_score_tokens", fail_score)
        chunks = [{"page": 2, "text": "beta"}, {"page": 1, "text": "alpha"}]
        ctx = app_module.build_context(chunks, question="alpha", max_chars=10_000)
        assert ctx.index("[Page 1]") < ctx.index("[Page 2]")


# ──────────────────────────────────────────────────────────────────────────────
# rechunk_text
# ──────────────────────────────────────────────────────────────────────────────

class TestRechunkText:
    def test_combines_small_pages(self, app_module):
        chunks = [{"page": i, "text": "x" * 100} for i in range(1, 6)]
        out = app_module.rechunk_text(chunks, target_chars=400)
        # 5 pages of 100 chars combined — should fit in fewer than 5 chunks
        assert len(out) < len(chunks)
        for piece in out:
            assert len(piece["text"]) <= 400 + 10  # tolerance for separators

    def test_splits_oversized_page(self, app_module):
        chunks = [{"page": 1, "text": "y" * 1000}]
        out = app_module.rechunk_text(chunks, target_chars=300)
        assert len(out) >= 4
        assert all(len(p["text"]) <= 300 for p in out)

    def test_preserves_total_content(self, app_module):
        original = "abcdefghij" * 50  # 500 chars
        chunks = [{"page": 1, "text": original}]
        out = app_module.rechunk_text(chunks, target_chars=120)
        joined = "".join(p["text"] for p in out)
        assert joined.replace("\n", "") == original.replace("\n", "")


# ──────────────────────────────────────────────────────────────────────────────
# performance regression checks
# ──────────────────────────────────────────────────────────────────────────────

class TestPerformanceOptimizations:
    def test_rechunk_large_pdf_is_fast_and_uses_larger_chunks(self, app_module):
        import time

        chunks = [{"page": i, "text": "x" * 1000} for i in range(1, 801)]

        start = time.perf_counter()
        pieces = app_module.rechunk_text(chunks, target_chars=app_module.CHUNK_CHAR_LIMIT)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0
        assert len(pieces) < 90

    def test_large_context_selection_is_fast(self, app_module):
        import time

        chunks = [
            {"page": i, "text": ("algoritma veri yapisi " if i % 17 == 0 else "alakasiz metin ") * 60}
            for i in range(1, 1201)
        ]

        start = time.perf_counter()
        ctx = app_module.build_context(
            chunks,
            question="algoritma veri yapisi nedir",
            max_chars=app_module.CHAT_CONTEXT_CHAR_LIMIT,
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0
        assert len(ctx) <= app_module.CHAT_CONTEXT_CHAR_LIMIT + 1200
        assert "algoritma" in ctx


# ──────────────────────────────────────────────────────────────────────────────
# _build_chat_prompt
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildChatPrompt:
    def test_tr_includes_history_block(self, app_module):
        app_module.st.session_state["lang"] = "tr"
        app_module.st.session_state["lang_selector"] = "tr"
        history = [
            {"q": "ilk soru", "a": "ilk cevap"},
            {"q": "ikinci soru", "a": "ikinci cevap"},
        ]
        prompt = app_module._build_chat_prompt("yeni soru", "PDF METNI", history)
        assert "Önceki konuşma" in prompt
        assert "ilk soru" in prompt
        assert "ikinci cevap" in prompt
        assert "PDF METNI" in prompt
        assert "yeni soru" in prompt

    def test_en_uses_english(self, app_module):
        app_module.st.session_state["lang"] = "en"
        app_module.st.session_state["lang_selector"] = "en"
        prompt = app_module._build_chat_prompt(
            "what is X?", "PDF BODY", history=[{"q": "foo", "a": "bar"}]
        )
        assert "Previous conversation" in prompt
        assert "what is X?" in prompt
        assert "PDF BODY" in prompt

    def test_history_capped_at_three(self, app_module):
        history = [{"q": f"q{i}", "a": f"a{i}"} for i in range(8)]
        prompt = app_module._build_chat_prompt("y", "ctx", history)
        # Only the last three should appear
        assert "q7" in prompt and "q6" in prompt and "q5" in prompt
        assert "q4" not in prompt and "q0" not in prompt

    def test_no_history_no_block(self, app_module):
        app_module.st.session_state["lang"] = "tr"
        app_module.st.session_state["lang_selector"] = "tr"
        prompt = app_module._build_chat_prompt("soru?", "ctx", [])
        assert "Önceki konuşma" not in prompt


# ──────────────────────────────────────────────────────────────────────────────
# extract_pdf_chunks
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractPDFChunks:
    def test_valid_pdf(self, app_module, make_pdf_bytes):
        data = make_pdf_bytes(["hello world", "second page"])
        chunks = app_module.extract_pdf_chunks(data)
        assert len(chunks) == 2
        assert chunks[0]["page"] == 1
        assert "hello" in chunks[0]["text"]

    def test_oversize_raises(self, app_module):
        too_big = b"x" * (app_module.MAX_PDF_BYTES + 1)
        with pytest.raises(app_module.PDFLoadError) as exc:
            app_module.extract_pdf_chunks(too_big)
        assert exc.value.key == "err_too_large"
        assert "limit" in exc.value.fmt

    def test_empty_pdf_raises_err_empty(self, app_module, make_pdf_bytes):
        data = make_pdf_bytes([""])  # one blank page, no text
        with pytest.raises(app_module.PDFLoadError) as exc:
            app_module.extract_pdf_chunks(data)
        assert exc.value.key == "err_empty"

    def test_garbage_bytes_raises_open_failed(self, app_module):
        with pytest.raises(app_module.PDFLoadError) as exc:
            app_module.extract_pdf_chunks(b"definitely not a pdf")
        assert exc.value.key == "err_open_failed"
        assert "detail" in exc.value.fmt

    def test_too_many_pages(self, app_module, make_pdf_bytes, monkeypatch):
        monkeypatch.setattr(app_module, "MAX_PDF_PAGES", 2)
        data = make_pdf_bytes(["a", "b", "c"])
        with pytest.raises(app_module.PDFLoadError) as exc:
            app_module.extract_pdf_chunks(data)
        assert exc.value.key == "err_too_many_pages"

    def test_encrypted_pdf(self, app_module, make_encrypted_pdf_bytes):
        data = make_encrypted_pdf_bytes("secret")
        with pytest.raises(app_module.PDFLoadError) as exc:
            app_module.extract_pdf_chunks(data)
        assert exc.value.key == "err_encrypted"


# ──────────────────────────────────────────────────────────────────────────────
# summary PDF export
# ──────────────────────────────────────────────────────────────────────────────

class TestSummaryPDFExport:
    def test_create_summary_pdf_bytes_keeps_turkish_text(self, app_module):
        import fitz

        summary = "# Başlık\n\nTürkçe özet şğıİ çığ"
        pdf_bytes = app_module.create_summary_pdf_bytes(summary, "Week 1.pdf")

        assert pdf_bytes.startswith(b"%PDF-")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        pix = doc[0].get_pixmap()
        bg_pixel = pix.pixel(8, 8)[:3]
        doc.close()

        assert "PDF Özeti" in text
        assert "Başlık" in text
        assert "Türkçe özet şğıİ çığ" in text
        assert max(bg_pixel) < 40

    def test_numbered_lists_are_preserved_inside_mixed_blocks(self, app_module):
        import fitz

        summary = """
OOP'nin Avantajları
OOP, prosedür tabanlı dillere göre üç önemli avantaj sunar:
1. Yönetilebilirlik: Kod büyüdükçe bakım kolaylaşır.
2. Veri Gizleme: Global verilere erişim kısıtlanır.
3. Gerçek Dünya Modelleme: Nesneler daha doğal simüle edilir.
4. Yeniden Kullanım: Sınıflar tekrar kullanılabilir.
""".strip()
        pdf_bytes = app_module.create_summary_pdf_bytes(summary, "Week 1.pdf")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()

        assert "1. Yönetilebilirlik" in text
        assert "2. Veri Gizleme" in text
        assert "3. Gerçek Dünya Modelleme" in text
        assert "4. Yeniden Kullanım" in text

    def test_headings_render_larger_than_body_text(self, app_module):
        import fitz

        pdf_bytes = app_module.create_summary_pdf_bytes(
            "# Büyük Başlık\n\nGövde metni burada yer alır.",
            "Week 1.pdf",
        )

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        sizes_by_text = {}
        for block in doc[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes_by_text[span["text"]] = span["size"]
        doc.close()

        assert sizes_by_text["Büyük Başlık"] > sizes_by_text["Gövde metni burada yer alır."] * 1.8

    def test_long_colon_intro_is_not_promoted_to_heading(self, app_module):
        assert app_module._is_summary_heading_line("Önemli Terimler:") is True
        assert app_module._is_summary_heading_line(
            "OOP, prosedür tabanlı dillere göre üç önemli avantaj sunar:"
        ) is False

    def test_summary_pdf_filename_is_safe_and_localized(self, app_module):
        assert app_module._summary_pdf_filename("Week 1 Algo.pdf") == "Week_1_Algo-ozet.pdf"

        app_module.st.session_state["lang"] = "en"
        app_module.st.session_state["lang_selector"] = "en"
        assert app_module._summary_pdf_filename("Week 1 Algo.pdf") == "Week_1_Algo-summary.pdf"

    def test_render_summary_tab_keeps_page_summary_and_adds_download(self, app_module, monkeypatch):
        downloads = []
        rendered = []

        app_module.st.session_state.summary = "Hazır özet"
        app_module.st.session_state.last_file_name = "Week 1.pdf"

        monkeypatch.setattr(
            app_module,
            "create_summary_pdf_bytes",
            lambda summary, file_name: b"%PDF-test",
        )
        monkeypatch.setattr(
            app_module.st,
            "download_button",
            lambda *args, **kwargs: downloads.append((args, kwargs)) or False,
        )
        monkeypatch.setattr(
            app_module.st,
            "markdown",
            lambda body=None, *_args, **_kwargs: rendered.append(body),
        )

        app_module.render_summary_tab()

        assert rendered[-1] == "Hazır özet"
        assert downloads
        assert downloads[0][0][0] == app_module.t("btn_download_summary")
        assert downloads[0][1]["data"] == b"%PDF-test"
        assert downloads[0][1]["file_name"] == "Week_1-ozet.pdf"
        assert downloads[0][1]["mime"] == "application/pdf"

    def test_render_summary_tab_reuses_cached_summary(self, app_module, monkeypatch):
        app_module.st.session_state.last_file_signature = "abc"
        app_module.st.session_state.last_file_name = "Week 1.pdf"
        app_module.st.session_state.summary_cache = {
            app_module.summary_cache_key(): "Cached summary"
        }

        monkeypatch.setattr(app_module.st, "button", lambda *_a, **_k: True)
        monkeypatch.setattr(
            app_module,
            "summarize_pdf",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should use cache")),
        )
        monkeypatch.setattr(app_module, "create_summary_pdf_bytes", lambda *_a, **_k: b"%PDF-test")
        monkeypatch.setattr(app_module.st, "download_button", lambda *_a, **_k: False)

        app_module.render_summary_tab()

        assert app_module.st.session_state.summary == "Cached summary"

    def test_render_summary_tab_reuses_cached_pdf_bytes(self, app_module, monkeypatch):
        downloads = []

        app_module.st.session_state.summary = "Cached PDF summary"
        app_module.st.session_state.last_file_signature = "abc"
        app_module.st.session_state.last_file_name = "Week 1.pdf"
        app_module.st.session_state.summary_pdf_cache = {
            app_module.summary_pdf_cache_key("Cached PDF summary"): b"%PDF-cached"
        }

        monkeypatch.setattr(
            app_module,
            "create_summary_pdf_bytes",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should use cached PDF")),
        )
        monkeypatch.setattr(
            app_module.st,
            "download_button",
            lambda *args, **kwargs: downloads.append((args, kwargs)) or False,
        )

        app_module.render_summary_tab()

        assert downloads[0][1]["data"] == b"%PDF-cached"


# ──────────────────────────────────────────────────────────────────────────────
# clean_text_for_tts — Web Speech API'sine vermeden önce markdown temizliği
# ──────────────────────────────────────────────────────────────────────────────


class TestCleanTextForTTS:
    def test_strips_bold_markers(self, app_module):
        out = app_module.clean_text_for_tts("**önemli** terim")
        assert "**" not in out
        assert "önemli" in out

    def test_strips_headings(self, app_module):
        out = app_module.clean_text_for_tts("# Başlık\n## Alt\n### Üst")
        assert "#" not in out
        assert "Başlık" in out
        assert "Alt" in out

    def test_strips_bullet_and_number_markers(self, app_module):
        out = app_module.clean_text_for_tts("- madde bir\n* madde iki\n1. madde üç")
        assert "- madde" not in out
        assert "* madde" not in out
        assert "1." not in out
        assert "madde bir" in out
        assert "madde üç" in out

    def test_strips_backticks(self, app_module):
        out = app_module.clean_text_for_tts("`kod` parçası")
        assert "`" not in out
        assert "kod" in out

    def test_collapses_excessive_newlines(self, app_module):
        out = app_module.clean_text_for_tts("a\n\n\n\nb")
        assert "\n\n\n" not in out

    def test_empty_input(self, app_module):
        assert app_module.clean_text_for_tts("") == ""
        assert app_module.clean_text_for_tts(None or "") == ""


# ──────────────────────────────────────────────────────────────────────────────
# ask_llm
# ──────────────────────────────────────────────────────────────────────────────

class TestAskLLM:
    def test_no_client_returns_error_prefix(self, app_module):
        app_module.client = None
        out = app_module.ask_llm("hi", "model")
        assert out.startswith(app_module.LLM_ERROR_PREFIX)

    def test_happy_path(self, app_module):
        _install_client(app_module, return_value=_resp("merhaba dünya"))
        out = app_module.ask_llm("hi", "model")
        assert out == "merhaba dünya"

    def test_passes_timeout_to_client(self, app_module):
        fake = _install_client(app_module, return_value=_resp("ok"))
        out = app_module.ask_llm("hi", "model", timeout_sec=7.5)
        assert out == "ok"
        assert fake.chat.completions.create.call_args.kwargs["timeout"] == 7.5

    def test_retries_then_succeeds(self, app_module):
        attempts = [
            Exception("rate_limit_exceeded; try again in 1s"),
            Exception("rate_limit_exceeded; try again in 1s"),
            _resp("ok"),
        ]
        fake = _install_client(app_module, side_effect=attempts)
        out = app_module.ask_llm("hi", "model")
        assert out == "ok"
        assert fake.chat.completions.create.call_count == 3

    def test_non_retryable_fails_fast(self, app_module):
        fake = _install_client(
            app_module, side_effect=[Exception("400 bad request: bogus")]
        )
        out = app_module.ask_llm("hi", "model")
        assert out.startswith(app_module.LLM_ERROR_PREFIX)
        assert fake.chat.completions.create.call_count == 1  # no retries

    def test_exhausts_retries(self, app_module):
        # Always rate-limited → eventually returns an error
        fake = _install_client(
            app_module,
            side_effect=Exception("429 rate_limit; try again in 1s"),
        )
        out = app_module.ask_llm("hi", "model")
        assert out.startswith(app_module.LLM_ERROR_PREFIX)
        assert fake.chat.completions.create.call_count == app_module.MAX_RETRIES


# ──────────────────────────────────────────────────────────────────────────────
# stream_llm
# ──────────────────────────────────────────────────────────────────────────────

class TestStreamLLM:
    def test_yields_deltas(self, app_module):
        _install_client(
            app_module, return_value=_stream(["He", "llo", " world"])
        )
        out = list(app_module.stream_llm("p", "m"))
        assert out == ["He", "llo", " world"]

    def test_passes_timeout_to_client(self, app_module):
        fake = _install_client(app_module, return_value=_stream(["ok"]))
        out = list(app_module.stream_llm("p", "m", timeout_sec=8.0))
        assert out == ["ok"]
        assert fake.chat.completions.create.call_args.kwargs["timeout"] == 8.0

    def test_skips_empty_deltas(self, app_module):
        _install_client(
            app_module, return_value=_stream(["a", None, "b", ""])
        )
        # None and "" deltas must be skipped (empty-string is falsy)
        assert list(app_module.stream_llm("p", "m")) == ["a", "b"]

    def test_no_client_yields_error(self, app_module):
        app_module.client = None
        out = list(app_module.stream_llm("p", "m"))
        assert len(out) == 1
        assert out[0].startswith(app_module.LLM_ERROR_PREFIX)

    def test_non_retryable_yields_error(self, app_module):
        _install_client(app_module, side_effect=[Exception("400 bad request")])
        out = list(app_module.stream_llm("p", "m"))
        assert out[-1].startswith(app_module.LLM_ERROR_PREFIX)

    def test_retry_then_stream(self, app_module):
        _install_client(
            app_module,
            side_effect=[
                Exception("rate_limit; try again in 1s"),
                _stream(["ok"]),
            ],
        )
        out = list(app_module.stream_llm("p", "m"))
        assert out == ["ok"]

    def test_partial_stream_error_does_not_retry_duplicate_answer(self, app_module):
        def broken_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))]
            )
            raise Exception("connection timeout")

        fake = _install_client(app_module, return_value=broken_stream())
        out = list(app_module.stream_llm("p", "m"))
        assert out[0] == "partial"
        assert out[1].startswith(app_module.LLM_ERROR_PREFIX)
        assert fake.chat.completions.create.call_count == 1


# ──────────────────────────────────────────────────────────────────────────────
# _stream_to_placeholder / chat UI behavior
# ──────────────────────────────────────────────────────────────────────────────

class TestStreamToPlaceholder:
    def test_shows_processing_then_final_answer(self, app_module):
        placeholder = _Placeholder()
        out = app_module._stream_to_placeholder(iter(["cev", "ap"]), placeholder)
        assert out == "cevap"
        assert placeholder.markdowns[0] == app_module.t("processing")
        assert placeholder.last == "cevap"
        assert placeholder.errors == []

    def test_error_before_tokens_renders_in_answer_slot(self, app_module):
        placeholder = _Placeholder()
        out = app_module._stream_to_placeholder(
            iter([f"{app_module.LLM_ERROR_PREFIX}rate limit"]),
            placeholder,
        )
        assert out == ""
        assert placeholder.errors == [app_module.t("llm_failed")]
        assert placeholder.last == app_module.t("llm_failed")

    def test_empty_stream_renders_empty_response_message(self, app_module):
        placeholder = _Placeholder()
        out = app_module._stream_to_placeholder(iter([]), placeholder)
        assert out == ""
        assert placeholder.errors == [app_module.t("llm_empty")]

    def test_can_suppress_errors_for_callers_with_fallbacks(self, app_module):
        placeholder = _Placeholder()
        out = app_module._stream_to_placeholder(
            iter([f"{app_module.LLM_ERROR_PREFIX}temporary"]),
            placeholder,
            show_errors=False,
        )
        assert out == ""
        assert placeholder.errors == []
        assert placeholder.empty_calls == 1

    def test_timer_estimate_is_shown_when_estimated_sec_given(self, app_module):
        placeholder = _Placeholder()
        out = app_module._stream_to_placeholder(
            iter(["bir ", "iki"]),
            placeholder,
            estimated_sec=12,
        )
        assert out == "bir iki"
        # İlk markdown başlangıç etiketini içermeli
        assert "12" in placeholder.markdowns[0]
        # En az bir token sırasında geri sayım göstergesi (⏱) görünmüş olmalı
        assert any("⏱" in (m or "") for m in placeholder.markdowns)
        # Final markdown sadece içerik (timer'sız)
        assert placeholder.last == "bir iki"

    def test_timer_disabled_when_estimated_sec_is_none(self, app_module):
        # Backwards-compat: chat tab timer'ı kapalı kullanır
        placeholder = _Placeholder()
        out = app_module._stream_to_placeholder(iter(["x"]), placeholder)
        assert out == "x"
        # processing etiketi olduğu gibi gösterilmeli (timer prefix yok)
        assert placeholder.markdowns[0] == app_module.t("processing")


class TestEstimateSummarySeconds:
    def test_small_pdf_gets_short_estimate(self, app_module):
        chunks = [{"page": 1, "text": "kısa içerik"}]
        secs = app_module.estimate_summary_seconds(chunks)
        assert 4 <= secs <= 20

    def test_huge_pdf_gets_chunked_estimate(self, app_module, monkeypatch):
        monkeypatch.setattr(app_module, "DIRECT_SUMMARY_CHAR_LIMIT", 1000)
        monkeypatch.setattr(app_module, "CHUNK_CHAR_LIMIT", 5000)
        chunks = [{"page": i, "text": "x" * 5000} for i in range(1, 21)]
        secs = app_module.estimate_summary_seconds(chunks)
        # 20 parça, 5 paralel → 4 batch; 4*4 + 8 = 24 sn civarı
        assert secs >= 10


class TestRenderChatTab:
    def test_submitted_question_is_streamed_stored_and_rerun(self, app_module, monkeypatch):
        placeholder = _Placeholder()
        reruns = []

        app_module.st.session_state.chunks = [{"page": 1, "text": "PDF text"}]
        app_module.st.session_state.chat_history = []

        def fake_stream(question, chunks, model_name, history=None):
            assert question == "programlama dilleri ne zaman kurulmuştur?"
            assert chunks == app_module.st.session_state.chunks
            assert history == []
            yield "cevap"

        monkeypatch.setattr(app_module, "chat_with_pdf_stream", fake_stream)
        monkeypatch.setattr(app_module.st, "text_input", lambda *_a, **_k: " programlama dilleri ne zaman kurulmuştur? ")
        monkeypatch.setattr(app_module.st, "form_submit_button", lambda *_a, **_k: True)
        monkeypatch.setattr(app_module.st, "empty", lambda: placeholder)
        monkeypatch.setattr(app_module.st, "rerun", lambda: reruns.append(True))

        app_module.render_chat_tab()

        assert app_module.st.session_state.chat_history == [
            {
                "q": "programlama dilleri ne zaman kurulmuştur?",
                "a": "cevap",
            }
        ]
        assert reruns == [True]

    def test_submitted_question_uses_chat_cache_without_streaming(self, app_module, monkeypatch):
        reruns = []
        question = "aynı soru"

        app_module.st.session_state.chunks = [{"page": 1, "text": "PDF text"}]
        app_module.st.session_state.chat_history = []
        cache_key = app_module.chat_cache_key(question, [])
        app_module.st.session_state.chat_cache = {cache_key: "cached answer"}

        monkeypatch.setattr(
            app_module,
            "chat_with_pdf_stream",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should use cache")),
        )
        monkeypatch.setattr(app_module.st, "text_input", lambda *_a, **_k: question)
        monkeypatch.setattr(app_module.st, "form_submit_button", lambda *_a, **_k: True)
        monkeypatch.setattr(app_module.st, "rerun", lambda: reruns.append(True))

        app_module.render_chat_tab()

        assert app_module.st.session_state.chat_history == [
            {"q": question, "a": "cached answer"}
        ]
        assert reruns == [True]


# ──────────────────────────────────────────────────────────────────────────────
# summarize_pdf
# ──────────────────────────────────────────────────────────────────────────────

class TestSummarizePDF:
    def test_medium_pdf_uses_single_direct_summary_call(self, app_module, monkeypatch):
        calls = []

        def fake_ask(prompt, model, max_tokens=2048):
            calls.append((prompt, max_tokens))
            return "Kapanış\nHızlı özet tam bir cümleyle bitti."

        monkeypatch.setattr(app_module, "ask_llm", fake_ask)

        chunks = [{"page": page, "text": "programlama " * 120} for page in range(1, 11)]
        out = app_module.summarize_pdf(chunks, "m")

        assert out == "Kapanış\nHızlı özet tam bir cümleyle bitti."
        assert len(calls) == 1
        assert calls[0][1] == app_module.SUMMARY_FINAL_MAX_TOKENS
        assert "PDF metni:" in calls[0][0]

    def test_retryable_direct_summary_falls_back_to_chunk_pipeline(self, app_module, monkeypatch):
        calls = []
        responses = iter([
            f"{app_module.LLM_ERROR_PREFIX}429 rate_limit",
            "chunk summary",
        ])

        def fake_ask(prompt, model, max_tokens=2048):
            calls.append(prompt)
            return next(responses)

        monkeypatch.setattr(app_module, "ask_llm", fake_ask)
        monkeypatch.setattr(app_module, "REQUEST_PACING_SEC", 0)

        chunks = [{"page": 1, "text": "x" * 5000}]
        out = app_module.summarize_pdf(chunks, "m")

        assert out == "chunk summary"
        assert len(calls) == 2

    def test_incomplete_final_summary_gets_continuation(self, app_module, monkeypatch):
        calls = []
        responses = iter([
            "chunk 1",
            "chunk 2",
            "1. kapsülleme\n2",
            "2. soyutlama\n\nKapanış\nAna çıkarımlar tamamlandı.",
        ])

        def fake_ask(prompt, model, max_tokens=2048):
            calls.append((prompt, max_tokens))
            return next(responses)

        monkeypatch.setattr(app_module, "ask_llm", fake_ask)
        monkeypatch.setattr(app_module, "REQUEST_PACING_SEC", 0)
        monkeypatch.setattr(app_module, "CHUNK_CHAR_LIMIT", 5500)
        monkeypatch.setattr(app_module, "DIRECT_SUMMARY_CHAR_LIMIT", 0)
        # Sıralı çalıştır — fake_ask iterator paylaşımı thread-safe değil.
        monkeypatch.setattr(app_module, "MAX_PARALLEL_CHUNK_REQUESTS", 1)

        chunks = [
            {"page": 1, "text": "a" * 5000},
            {"page": 2, "text": "b" * 5000},
        ]
        out = app_module.summarize_pdf(chunks, "m")

        assert out.endswith("Ana çıkarımlar tamamlandı.")
        assert len(calls) == 4
        assert calls[-1][1] == app_module.SUMMARY_CONTINUATION_MAX_TOKENS

    def test_complete_final_summary_is_not_continued(self, app_module, monkeypatch):
        calls = []

        def fake_ask(prompt, model, max_tokens=2048):
            calls.append(max_tokens)
            if len(calls) <= 2:
                return f"chunk {len(calls)}"
            return "Kapanış\nÖzet tam bir cümleyle bitti."

        monkeypatch.setattr(app_module, "ask_llm", fake_ask)
        monkeypatch.setattr(app_module, "REQUEST_PACING_SEC", 0)
        monkeypatch.setattr(app_module, "CHUNK_CHAR_LIMIT", 5500)
        monkeypatch.setattr(app_module, "DIRECT_SUMMARY_CHAR_LIMIT", 0)
        # Sıralı çalıştır — len(calls) tabanlı state thread-safe değil.
        monkeypatch.setattr(app_module, "MAX_PARALLEL_CHUNK_REQUESTS", 1)

        chunks = [
            {"page": 1, "text": "a" * 5000},
            {"page": 2, "text": "b" * 5000},
        ]
        out = app_module.summarize_pdf(chunks, "m")

        assert out == "Kapanış\nÖzet tam bir cümleyle bitti."
        assert calls == [
            app_module.SUMMARY_CHUNK_MAX_TOKENS,
            app_module.SUMMARY_CHUNK_MAX_TOKENS,
            app_module.SUMMARY_FINAL_MAX_TOKENS,
        ]

    def test_partial_failure_still_returns_summary(self, app_module, monkeypatch):
        # 3 chunks → first fails, others succeed; final merge succeeds.
        responses = iter([
            f"{app_module.LLM_ERROR_PREFIX}rate limit",  # chunk 1 fails
            "summary 2",                                  # chunk 2 ok
            "summary 3",                                  # chunk 3 ok
            "FINAL MERGED",                               # final combine
        ])

        def fake_ask(prompt, model, max_tokens=2048):
            return next(responses)

        monkeypatch.setattr(app_module, "ask_llm", fake_ask)
        monkeypatch.setattr(app_module, "REQUEST_PACING_SEC", 0)
        # Big enough that rechunk_text doesn't split a single 5000-char page,
        # small enough that the three pages stay separate pieces.
        monkeypatch.setattr(app_module, "CHUNK_CHAR_LIMIT", 5500)
        monkeypatch.setattr(app_module, "DIRECT_SUMMARY_CHAR_LIMIT", 0)
        # iter() üzerinde paylaşılan state — sıralı çalıştır.
        monkeypatch.setattr(app_module, "MAX_PARALLEL_CHUNK_REQUESTS", 1)

        chunks = [
            {"page": 1, "text": "a" * 5000},
            {"page": 2, "text": "b" * 5000},
            {"page": 3, "text": "c" * 5000},
        ]
        out = app_module.summarize_pdf(chunks, "m")
        assert out == "FINAL MERGED"

    def test_all_chunks_fail_returns_error(self, app_module, monkeypatch):
        def fake_ask(prompt, model, max_tokens=2048):
            return f"{app_module.LLM_ERROR_PREFIX}down"

        monkeypatch.setattr(app_module, "ask_llm", fake_ask)
        monkeypatch.setattr(app_module, "REQUEST_PACING_SEC", 0)

        chunks = [{"page": 1, "text": "x" * 5000}]
        out = app_module.summarize_pdf(chunks, "m")
        assert out.startswith(app_module.LLM_ERROR_PREFIX)

    def test_single_partial_skips_final_merge(self, app_module, monkeypatch):
        # One chunk only → no second LLM call needed; we should get the partial back.
        calls = []

        def fake_ask(prompt, model, max_tokens=2048):
            calls.append(prompt)
            return "only-summary"

        monkeypatch.setattr(app_module, "ask_llm", fake_ask)
        monkeypatch.setattr(app_module, "REQUEST_PACING_SEC", 0)

        chunks = [{"page": 1, "text": "short"}]
        out = app_module.summarize_pdf(chunks, "m")
        assert out == "only-summary"
        assert len(calls) == 1


# ──────────────────────────────────────────────────────────────────────────────
# stream_summarize_pdf — yeni streaming yolu
# ──────────────────────────────────────────────────────────────────────────────


class TestStreamSummarizePDF:
    def test_small_pdf_streams_directly_via_stream_llm(self, app_module, monkeypatch):
        captured = {}

        def fake_stream(prompt, model, max_tokens=2048):
            captured["prompt"] = prompt
            captured["max_tokens"] = max_tokens
            yield "Bölüm "
            yield "1: "
            yield "Hızlı özet."

        # ask_llm asla çağrılmamalı — stream yolu doğrudan akıtmalı.
        def fail_ask(*_a, **_k):
            raise AssertionError("direct path must not call ask_llm")

        monkeypatch.setattr(app_module, "stream_llm", fake_stream)
        monkeypatch.setattr(app_module, "ask_llm", fail_ask)

        chunks = [{"page": 1, "text": "kısa içerik"}]
        out = "".join(app_module.stream_summarize_pdf(chunks, "m"))

        assert out == "Bölüm 1: Hızlı özet."
        assert "PDF metni:" in captured["prompt"]
        assert captured["max_tokens"] == app_module.SUMMARY_FINAL_MAX_TOKENS

    def test_chunked_pdf_streams_only_final_pass(self, app_module, monkeypatch):
        ask_calls = []
        stream_calls = []

        def fake_ask(prompt, model, max_tokens=2048):
            ask_calls.append(max_tokens)
            return f"ara özet {len(ask_calls)}"

        def fake_stream(prompt, model, max_tokens=2048):
            stream_calls.append(max_tokens)
            yield "FINAL"

        monkeypatch.setattr(app_module, "ask_llm", fake_ask)
        monkeypatch.setattr(app_module, "stream_llm", fake_stream)
        monkeypatch.setattr(app_module, "CHUNK_CHAR_LIMIT", 5500)
        monkeypatch.setattr(app_module, "DIRECT_SUMMARY_CHAR_LIMIT", 0)
        monkeypatch.setattr(app_module, "MAX_PARALLEL_CHUNK_REQUESTS", 1)

        chunks = [
            {"page": 1, "text": "a" * 5000},
            {"page": 2, "text": "b" * 5000},
        ]
        out = "".join(app_module.stream_summarize_pdf(chunks, "m"))

        # 2 ara özet ask_llm üzerinden, son birleştirme stream_llm üzerinden
        assert len(ask_calls) == 2
        assert all(c == app_module.SUMMARY_CHUNK_MAX_TOKENS for c in ask_calls)
        assert stream_calls == [app_module.SUMMARY_FINAL_MAX_TOKENS]
        assert out == "FINAL"

    def test_single_partial_yields_without_extra_llm_call(self, app_module, monkeypatch):
        ask_calls = []
        stream_calls = []

        def fake_ask(prompt, model, max_tokens=2048):
            ask_calls.append(prompt)
            return "tek ara özet"

        def fake_stream(prompt, model, max_tokens=2048):
            stream_calls.append(prompt)
            yield "ASLA"

        monkeypatch.setattr(app_module, "ask_llm", fake_ask)
        monkeypatch.setattr(app_module, "stream_llm", fake_stream)
        monkeypatch.setattr(app_module, "DIRECT_SUMMARY_CHAR_LIMIT", 0)
        monkeypatch.setattr(app_module, "MAX_PARALLEL_CHUNK_REQUESTS", 1)

        chunks = [{"page": 1, "text": "kısa"}]
        out = "".join(app_module.stream_summarize_pdf(chunks, "m"))

        assert out == "tek ara özet"
        assert len(ask_calls) == 1
        assert stream_calls == []  # tek parça → ekstra streaming çağrısı yok

    def test_all_chunks_fail_yields_error_prefix(self, app_module, monkeypatch):
        def fake_ask(prompt, model, max_tokens=2048):
            return f"{app_module.LLM_ERROR_PREFIX}rate limit"

        monkeypatch.setattr(app_module, "ask_llm", fake_ask)
        monkeypatch.setattr(app_module, "DIRECT_SUMMARY_CHAR_LIMIT", 0)
        monkeypatch.setattr(app_module, "MAX_PARALLEL_CHUNK_REQUESTS", 1)

        chunks = [{"page": 1, "text": "x" * 5000}]
        out = list(app_module.stream_summarize_pdf(chunks, "m"))

        assert len(out) == 1
        assert out[0].startswith(app_module.LLM_ERROR_PREFIX)


class TestSummarizeChunksInParallel:
    def test_parallel_run_preserves_order_and_collects_results(self, app_module, monkeypatch):
        # Paralel çalışsa bile sıra korunmalı — parça i'nin özeti sonuçta
        # i'inci pozisyonda olur.
        import threading

        seen_threads = set()

        def fake_ask(prompt, model, max_tokens=2048):
            seen_threads.add(threading.get_ident())
            # prompt içindeki id'yi geri ver — sırayı doğrulamak için
            for marker in ("ALPHA", "BETA", "GAMMA"):
                if marker in prompt:
                    return f"summary-{marker}"
            return "summary-?"

        monkeypatch.setattr(app_module, "ask_llm", fake_ask)
        monkeypatch.setattr(app_module, "MAX_PARALLEL_CHUNK_REQUESTS", 3)

        pieces = [
            {"text": "ALPHA content"},
            {"text": "BETA content"},
            {"text": "GAMMA content"},
        ]
        partials, failures = app_module._summarize_chunks_in_parallel(
            pieces, "{text}", "m"
        )

        assert partials == ["summary-ALPHA", "summary-BETA", "summary-GAMMA"]
        assert failures == 0
        # En az iki ayrı thread'ten çağrı gelmiş olmalı (paralel çalışma kanıtı)
        assert len(seen_threads) >= 2

    def test_failures_are_excluded_and_counted(self, app_module, monkeypatch):
        from threading import Lock
        lock = Lock()
        counter = {"n": 0}

        def fake_ask(prompt, model, max_tokens=2048):
            with lock:
                counter["n"] += 1
                idx = counter["n"]
            if idx == 2:
                return f"{app_module.LLM_ERROR_PREFIX}429"
            return f"ok-{idx}"

        monkeypatch.setattr(app_module, "ask_llm", fake_ask)
        monkeypatch.setattr(app_module, "MAX_PARALLEL_CHUNK_REQUESTS", 1)

        pieces = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        partials, failures = app_module._summarize_chunks_in_parallel(
            pieces, "{text}", "m"
        )

        assert failures == 1
        assert len(partials) == 2


# ──────────────────────────────────────────────────────────────────────────────
# chat_with_pdf / chat_with_pdf_stream end-to-end
# ──────────────────────────────────────────────────────────────────────────────

class TestChatWrappers:
    def test_chat_with_pdf_uses_relevant_context(self, app_module):
        captured = {}

        def fake_ask(prompt, model, max_tokens=2048):
            captured["prompt"] = prompt
            return "answer"

        app_module.ask_llm = fake_ask  # type: ignore[attr-defined]

        chunks = [
            {"page": 1, "text": "alakasiz icerik"},
            {"page": 2, "text": "elma fiyatlari yuksek"},
        ]
        out = app_module.chat_with_pdf("elma fiyatlari nedir", chunks, "m")
        assert out == "answer"
        assert "elma fiyatlari yuksek" in captured["prompt"]

    def test_stream_wrapper_uses_history(self, app_module):
        captured = {}

        def fake_stream(prompt, model, max_tokens=2048):
            captured["prompt"] = prompt
            yield "ok"

        app_module.stream_llm = fake_stream  # type: ignore[attr-defined]

        history = [{"q": "önceki", "a": "yanıt"}]
        gen = app_module.chat_with_pdf_stream("yeni", [], "m", history=history)
        assert list(gen) == ["ok"]
        assert "önceki" in captured["prompt"]
