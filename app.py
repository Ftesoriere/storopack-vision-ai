import streamlit as st
import re
import json
from datetime import datetime

st.set_page_config(
    page_title="Storopack Vision AI Analyzer",
    page_icon="📦",
    layout="wide"
)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'current_analysis' not in st.session_state:
    st.session_state['current_analysis'] = None
if 'saved_api_key' not in st.session_state:
    st.session_state['saved_api_key'] = ""

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
    <div style="background: linear-gradient(135deg, #003366, #0056b3); padding: 22px 28px; border-radius: 12px; color: white; margin-bottom: 25px;">
        <h1 style="margin:0; font-size: 26px; font-weight: 700;">📦 Storopack Vision AI Analyzer</h1>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size: 14px;">
            Analisi nativa del flusso video YouTube tramite Google Gemini &mdash; rilevamento di PAPERplus&reg;, AIRplus&reg;, FOAMplus&reg;, PELASPAN&reg;
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR - API KEY
# ---------------------------------------------------------------------------
st.sidebar.header("🔑 Google AI Studio API Key")

default_key = st.session_state.get('saved_api_key', '')
try:
    if not default_key and "GEMINI_API_KEY" in st.secrets:
        default_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

input_key = st.sidebar.text_input(
    "API Key (formato AIza...):",
    value=default_key,
    type="password",
    help="Genera la chiave su https://aistudio.google.com/apikey. Deve iniziare con 'AIza'."
)

if input_key:
    st.session_state['saved_api_key'] = input_key.strip()

active_api_key = st.session_state.get('saved_api_key', '')

# Validazione formato chiave
key_format_ok = active_api_key.startswith("AIza") and len(active_api_key) > 30

if not active_api_key:
    st.sidebar.warning("🔴 Nessuna API Key inserita.")
elif not key_format_ok:
    st.sidebar.error(
        "🔴 Formato chiave non valido.\n\n"
        "Stai usando una credenziale OAuth / Service Account. "
        "Serve una API Key di **Google AI Studio** che inizia con `AIza`."
    )
    st.sidebar.markdown("👉 [Genera la chiave corretta](https://aistudio.google.com/apikey)")
else:
    st.sidebar.success("🟢 API Key AI Studio valida e salvata in sessione")

st.sidebar.divider()

st.sidebar.header("⚙️ Parametri di Analisi")
model_choice = st.sidebar.selectbox(
    "Modello Gemini:",
    ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
    index=0
)
fps_choice = st.sidebar.select_slider(
    "Campionamento temporale (fps):",
    options=[0.2, 0.5, 1.0, 2.0],
    value=1.0,
    help="1.0 = un frame al secondo. Valori più alti aumentano precisione e costo."
)

st.sidebar.header("🎯 Materiali da cercare")
targets = {
    "PAPERplus": st.sidebar.checkbox("🟤 PAPERplus® / PAPERwrap (carta)", value=True),
    "AIRplus": st.sidebar.checkbox("🎈 AIRplus® / AIRmove² (cuscini d'aria)", value=True),
    "FOAMplus": st.sidebar.checkbox("🔲 FOAMplus® (schiuma)", value=True),
    "PELASPAN": st.sidebar.checkbox("⚪ PELASPAN® (chip loose fill)", value=True),
    "CARTONE": st.sidebar.checkbox("📦 Scatole / nastro adesivo", value=True),
}

st.sidebar.header("📜 Cronologia")
if st.session_state['history']:
    for idx, h in enumerate(st.session_state['history']):
        if st.sidebar.button(f"▶ {h['video_id']} ({len(h['results'])} esiti)", key=f"hist_{idx}"):
            st.session_state['current_analysis'] = h
else:
    st.sidebar.caption("Nessun video analizzato in questa sessione.")

# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------
def extract_youtube_id(url):
    if not url:
        return None
    clean = url.strip()
    m = re.search(r'(?:youtu\.be/|youtube\.com/(?:embed/|v/|watch\?v=|watch\?.+&v=))([\w-]{11})', clean)
    if m:
        return m.group(1)
    if len(clean) == 11 and '/' not in clean:
        return clean
    return None


def build_prompt(active_targets, fps):
    wanted = [k for k, v in active_targets.items() if v]
    return f"""Sei un ispettore tecnico specializzato in imballaggi protettivi industriali Storopack.

Guarda l'INTERO video e individua OGNI momento in cui compare o viene manipolato
materiale da imballaggio protettivo. Categorie da cercare: {', '.join(wanted)}.

Riferimenti visivi:
- PAPERplus / PAPERwrap: carta Kraft marrone o bianca, stropicciata, a fisarmonica,
  a nido d'ape, fasci o strisce di carta inserite nella scatola.
- AIRplus / AIRmove: catene di cuscini d'aria in plastica trasparente, film a bolle
  d'aria (pluriball), sacchetti gonfiati.
- FOAMplus: schiuma poliuretanica espansa, sacchetti di schiuma auto-modellanti,
  inserti in polistirolo bianco sagomato.
- PELASPAN: chip/trucioli sfusi a forma di S, bianchi o verdi.
- CARTONE: scatole in cartone ondulato, nastro adesivo da imballaggio, etichette.

REGOLE OBBLIGATORIE:
1. Fornisci il timestamp REALE osservato nel video, nel formato MM:SS.
2. Dai priorità alle AZIONI (operatore o macchina che inserisce/avvolge il materiale)
   rispetto alla semplice presenza passiva sullo sfondo.
3. Non inventare timestamp: se un materiale non compare, non elencarlo.
4. Se lo stesso materiale resta visibile a lungo, indica una sola riga con il momento
   più rappresentativo.
5. La confidenza deve riflettere quanto sei realmente sicuro (0-100).

Campionamento richiesto: circa {fps} frame al secondo.

Rispondi ESCLUSIVAMENTE con un array JSON valido, senza testo prima o dopo:
[
  {{
    "time": "00:12",
    "seconds": 12,
    "title": "Inserimento fascio di carta Kraft",
    "description": "L'operatore inserisce a due mani una striscia di carta Kraft marrone nella scatola di cartone aperta sul banco.",
    "confidence": "94%",
    "category": "PAPERplus",
    "action": true
  }}
]
Se non rilevi alcun materiale da imballaggio, rispondi con: []
"""


def analyze_youtube_native(api_key, youtube_url, model_name, prompt, fps):
    """
    Invia l'URL YouTube direttamente a Gemini: il video viene elaborato
    lato Google, senza download locale.
    Ritorna: (results, diagnostics)
    """
    diagnostics = {
        "sdk": None,
        "model_requested": model_name,
        "method": "native_youtube_uri",
        "raw_response_excerpt": None,
        "error": None,
    }

    try:
        from google import genai
        from google.genai import types
        diagnostics["sdk"] = "google-genai"
    except ImportError as e:
        diagnostics["error"] = f"SDK google-genai non disponibile: {e}"
        return None, diagnostics

    try:
        client = genai.Client(api_key=api_key)

        video_part = types.Part(
            file_data=types.FileData(file_uri=youtube_url),
            video_metadata=types.VideoMetadata(fps=fps),
        )

        response = client.models.generate_content(
            model=model_name,
            contents=types.Content(parts=[video_part, types.Part(text=prompt)]),
        )

        text = (response.text or "").strip()
        diagnostics["raw_response_excerpt"] = text[:600]

        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            diagnostics["error"] = "Nessun array JSON trovato nella risposta del modello."
            return None, diagnostics

        parsed = json.loads(match.group(0))
        for item in parsed:
            item["engine_used"] = f"Gemini nativo su URL YouTube ({model_name})"
        return parsed, diagnostics

    except Exception as e:
        diagnostics["error"] = f"{type(e).__name__}: {e}"

        # Secondo tentativo senza video_metadata (alcuni modelli non la accettano)
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=types.Content(parts=[
                    types.Part(file_data=types.FileData(file_uri=youtube_url)),
                    types.Part(text=prompt),
                ]),
            )
            text = (response.text or "").strip()
            diagnostics["raw_response_excerpt"] = text[:600]
            diagnostics["method"] = "native_youtube_uri_fallback_no_fps"
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                for item in parsed:
                    item["engine_used"] = f"Gemini nativo su URL YouTube ({model_name}, fallback)"
                diagnostics["error"] = None
                return parsed, diagnostics
        except Exception as e2:
            diagnostics["error"] = f"{diagnostics['error']} | Fallback: {type(e2).__name__}: {e2}"

        return None, diagnostics


# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Video da analizzare")
    video_url = st.text_input(
        "URL YouTube:",
        value="https://www.youtube.com/watch?v=Uo3pD0sRbII"
    )
    btn = st.button("🚀 Analizza video con Gemini", type="primary", disabled=not key_format_ok)

    if not key_format_ok:
        st.caption("⚠️ Inserisci una API Key valida di Google AI Studio per abilitare l'analisi.")

    video_id = extract_youtube_id(video_url)
    if video_id:
        st.video(f"https://www.youtube.com/watch?v={video_id}")

with col_right:
    st.subheader("2. Materiali rilevati")

    if btn and video_id and key_format_ok:
        clean_url = f"https://www.youtube.com/watch?v={video_id}"
        prompt = build_prompt(targets, fps_choice)

        with st.spinner(f"Gemini sta guardando l'intero video a {fps_choice} fps… (30-90 s per video lunghi)"):
            results, diag = analyze_youtube_native(
                active_api_key, clean_url, model_choice, prompt, fps_choice
            )

        if results is None:
            st.error("❌ Analisi non riuscita. Dettagli tecnici qui sotto.")
            with st.expander("🔧 Diagnostica tecnica", expanded=True):
                st.json(diag)
        elif len(results) == 0:
            st.warning("Gemini ha analizzato il video ma non ha rilevato materiale da imballaggio.")
            with st.expander("🔧 Diagnostica tecnica"):
                st.json(diag)
        else:
            analysis_obj = {
                "video_id": video_id,
                "youtube_url": clean_url,
                "results": results,
                "diagnostics": diag,
                "model": model_choice,
                "fps": fps_choice,
                "analyzed_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
            st.session_state['current_analysis'] = analysis_obj
            st.session_state['history'] = [
                h for h in st.session_state['history'] if h['video_id'] != video_id
            ] + [analysis_obj]

    # -----------------------------------------------------------------------
    # RENDER RISULTATI
    # -----------------------------------------------------------------------
    current = st.session_state.get('current_analysis')
    if current and current.get('results'):
        results = current['results']
        vid = current['video_id']

        st.success(
            f"✅ {len(results)} rilevamenti su `{vid}` — "
            f"modello `{current.get('model')}` a `{current.get('fps')}` fps"
        )

        audit = {
            "storopack_vision_ai_audit": {
                "generated_at": current.get("analyzed_at_utc"),
                "video_metadata": {
                    "video_id": vid,
                    "youtube_url": current.get("youtube_url"),
                    "model": current.get("model"),
                    "sampling_fps": current.get("fps"),
                },
                "api_diagnostics": current.get("diagnostics"),
                "detection_findings": results,
            }
        }

        st.download_button(
            "📥 Scarica report di audit (JSON)",
            data=json.dumps(audit, indent=2, ensure_ascii=False),
            file_name=f"storopack_audit_{vid}.json",
            mime="application/json",
        )

        st.divider()

        for item in results:
            icon = "🎬" if item.get("action") else "👁️"
            st.markdown(f"### {icon} {item.get('time', '??:??')} — {item.get('title', 'Materiale')}")
            st.write(item.get("description", ""))
            c1, c2 = st.columns(2)
            c1.metric("Confidenza", item.get("confidence", "n/d"))
            c2.metric("Categoria", item.get("category", "n/d"))
            secs = item.get("seconds", 0)
            st.markdown(
                f"[↗ Apri il video al secondo {secs}]"
                f"(https://www.youtube.com/watch?v={vid}&t={secs}s)"
            )
            st.divider()

        with st.expander("🔧 Diagnostica tecnica della chiamata"):
            st.json(current.get("diagnostics", {}))
    elif not btn:
        st.info("Inserisci un URL YouTube e avvia l'analisi.")
