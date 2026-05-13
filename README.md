# PDF Assistant — mtk.

AI tabanlı PDF özetleme ve sohbet asistanı. PDF yükle, anında özet çıkar, içeriğe sorular sor, özeti sesli dinle.

Streamlit + Groq LLM API ile yazıldı. Türkçe ve İngilizce destekler.

## Özellikler

- **Hızlı özet üretimi** — paralel chunk işleme + streaming çıktı; tipik 28 sayfalık PDF için ~10-15 sn
- **PDF ile sohbet** — yüklediğin PDF'in içeriğine sorular sor, llama-3.3-70b-versatile yanıtlar
- **Akıllı fallback** — 8B-instant başarısız olursa otomatik 70B-versatile devreye girer
- **Canlı geri sayım** — özet üretilirken tahmini süre + elapsed sayaç
- **Sesli okuma** — tarayıcının yerleşik speechSynthesis API'siyle özeti dinleyebilirsin (ücretsiz, gecikmesiz)
- **Stilli PDF dışa aktarım** — özeti şık formatlı PDF olarak indir
- **İki dil** — Türkçe / İngilizce toggle
- **Önbellek** — aynı PDF + aynı dil için tekrar tıklamada anında sonuç

## Kurulum

### 1. Repo'yu klonla

```bash
git clone https://github.com/<kullanici>/<repo>.git
cd <repo>
```

### 2. Sanal ortam oluştur ve aktif et

**Windows (PowerShell):**
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 4. Groq API anahtarını ayarla

`.env.example`'ı `.env` olarak kopyala ve [console.groq.com/keys](https://console.groq.com/keys) adresinden aldığın anahtarı yaz:

```bash
cp .env.example .env
# .env dosyasını editle, GROQ_API_KEY=... satırını doldur
```

### 5. Çalıştır

```bash
python -m streamlit run app.py
```

Tarayıcı otomatik `http://localhost:8501` açar.

## Kullanım

1. Sol kenar çubuğundan PDF yükle (maks 60 MB, 800 sayfa)
2. **Özet oluştur** butonuna bas → birkaç saniye içinde streaming olarak özet gelir
3. **Sohbet** sekmesinden PDF içeriği hakkında soru sor
4. Özeti **PDF indir** butonuyla stilli formatta indir, veya **Sesli oku** ile dinle

## Test

```bash
pytest tests/ -q
```

97 test (retry mantığı, RAG seçimi, chunking, özet pipeline'ı, streaming, TTS markdown temizliği).

## Mimari notlar

- **Özet pipeline'ı (`stream_summarize_pdf`)**: PDF < 90k karakter ise direct yol (tek streaming LLM çağrısı); büyükse rechunk → 5-worker paralel ara özetler → hiyerarşik birleştirme → streaming final.
- **Model stratejisi**: Özet için `llama-3.1-8b-instant` (hız), başarısız olursa `llama-3.3-70b-versatile` (kalite). Sohbet için her zaman 70B-versatile.
- **Hata toleransı**: 413/429/5xx için exponential backoff + jitter; tek çağrı için 45 sn üst sınır.
- **Önbellekler**: `summary_cache`, `summary_pdf_cache`, `chat_cache` — hepsi PDF imzası + dil ile anahtarlanır.
- **Başarısız model belleği**: Bir model session'da hata verirse aynı session'da tekrar denenmez.

## Bağımlılıklar

- `streamlit` — web arayüzü
- `PyMuPDF (fitz)` — PDF metin çıkarımı
- `groq` — LLM API client
- `python-dotenv` — `.env` yükleme
- `reportlab` — stilli PDF çıktısı

## Lisans

[MIT](LICENSE) — özgürce kullan, fork'la, değiştir.
