"""Stub Streamlit so app.py can be imported in a plain Python process.

app.py touches `st.set_page_config`, `st.session_state`, `st.markdown`, etc. at
import time and inside helpers we want to test. We replace `streamlit` with a
deliberately permissive shim before app is imported.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _SessionState(dict):
    """Behaves like both attr-bag and dict — what Streamlit actually exposes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Progress:
    def progress(self, *_a, **_k):
        pass

    def empty(self):
        pass


class _Empty:
    def __init__(self):
        self.last = None
        self.errors = []

    def markdown(self, body=None, *_a, **_k):
        self.last = body

    def error(self, body=None, *_a, **_k):
        self.errors.append(body)
        self.last = body

    def empty(self):
        self.last = None


def _build_streamlit_stub() -> types.ModuleType:
    st = types.ModuleType("streamlit")

    st.session_state = _SessionState()

    def _noop(*_a, **_k):
        return None

    # Top-of-file calls
    st.set_page_config = _noop
    st.set_option = _noop

    # Layout / output
    st.markdown = _noop
    st.error = _noop
    st.warning = _noop
    st.info = _noop
    st.success = _noop
    st.write = _noop
    st.rerun = _noop
    st.stop = _noop

    # Containers / context managers
    st.spinner = lambda *a, **k: _Ctx()
    st.form = lambda *a, **k: _Ctx()
    st.sidebar = _Ctx()
    st.expander = lambda *a, **k: _Ctx()

    # Widgets
    st.button = lambda *a, **k: False
    st.form_submit_button = lambda *a, **k: False
    st.download_button = lambda *a, **k: False
    st.text_input = lambda *a, **k: ""
    st.text_area = lambda *a, **k: ""
    st.radio = lambda *a, **k: "tr"
    st.file_uploader = lambda *a, **k: None
    st.tabs = lambda labels: [_Ctx() for _ in labels]
    st.columns = lambda spec, **k: [_Ctx() for _ in (spec if isinstance(spec, (list, tuple)) else range(spec))]
    st.empty = lambda: _Empty()
    st.progress = lambda *a, **k: _Progress()

    return st


def _install_streamlit_stub() -> None:
    if "streamlit" in sys.modules and not isinstance(
        sys.modules["streamlit"], types.ModuleType
    ):
        return

    st = _build_streamlit_stub()
    sys.modules["streamlit"] = st

    components = types.ModuleType("streamlit.components")
    components_v1 = types.ModuleType("streamlit.components.v1")

    def _html(*_a, **_k):
        return None

    components_v1.html = _html
    components.v1 = components_v1
    sys.modules["streamlit.components"] = components
    sys.modules["streamlit.components.v1"] = components_v1


_install_streamlit_stub()
# Avoid pulling a real .env into the test process.
os.environ.pop("GROQ_API_KEY", None)


import pytest  # noqa: E402  (imported after path setup intentionally)


@pytest.fixture
def app_module():
    """Fresh import of app.py with the stub already in place."""
    sys.modules.pop("app", None)
    import app  # type: ignore  # noqa: WPS433

    # Each test starts with the default language and a clean session.
    app.st.session_state.clear()
    app.st.session_state["lang"] = "tr"
    app.st.session_state["lang_selector"] = "tr"
    return app


@pytest.fixture
def make_pdf_bytes():
    """Build a minimal in-memory PDF with the given page texts."""
    import fitz

    def _make(pages):
        doc = fitz.open()
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        data = doc.tobytes()
        doc.close()
        return data

    return _make


@pytest.fixture
def make_encrypted_pdf_bytes(tmp_path):
    """Build an AES-256 encrypted PDF and return its bytes."""
    import fitz

    def _make(text="hidden"):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), text)
        out = tmp_path / "enc.pdf"
        doc.save(
            str(out),
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
        )
        doc.close()
        return out.read_bytes()

    return _make
