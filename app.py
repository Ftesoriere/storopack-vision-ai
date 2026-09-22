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
for key, default in [
    ('history', []),
    ('current_analysis', None),
    ('saved_api_key', ""),
    ('key_validated', False),
    ('key_validation_msg', ""),
    ('available_models', []),
    ('autocheck_done', False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Modelli di ripiego se la lista live non è ancora stata caricata.
# Ordine: dal più recente al più vecchio.
FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.6-pro",
    "gemini-3.5-flash",
    "gemini-3.0-flash",
    "gemini-2.0-flash",
]

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
    <div style="background: linear-gradient(135deg, #003366, #0056b3); padding: 22px 28px; border-radius: 12px; color: white; margin-bottom: 25px;">
        <h1 style="margin:0; font-size: 26px; font-weight: 700;">📦 Storopack Vision AI Analyzer</h1>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size: 14px;">
            Analisi nativa del flusso video YouTube tramite Google Gemini &mdash; PAPERplus&reg;, AIRplus&reg;, FOAMplus&reg;, PELASPAN&reg;
        </p>
    </div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# VALIDAZIONE CHIAVE
# ---------------------------------------------------------------------------
def validate_key_live(api_key):
    """Interroga l'endpoint Gemini e restituisce i modelli realmente disponibili."""
    try:
        from google import genai
    except ImportError as e:
        return False, f"SDK google-genai non installato: {e}", []

    try:
        client = genai.Client(api_key=api_key)
        models = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            short = name.replace("models/", "")
            if not short:
                continue
            actions = getattr(m, "supported_actions", None) or []
            if actions and "generateContent" not in actions:
                continue
            # Scarta embedding, imagen, veo, tts: non servono per l'analisi video
            if any(x in short for x in ("embedding", "imagen", "veo", "tts", "aqa")):
                continue
            models.append(short)

        if models:
            # Ordina: più recenti prima
            models.sort(key=lambda s: (
                "3.6" not in s, "3.5" not in s, "flash" not in s, s
            ))
            return True, f"Connessione riuscita — {len(models)} modelli disponibili", models
        return False, "Connessione riuscita ma nessun modello generativo compatibile", []
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", []


# ---------------------------------------------------------------------------
# SIDEBAR - API KEY
# ---------------------------------------------------------------------------
st.sidebar.header("🔑 Gemini API Key")

default_key = st.session_state.get('saved_api_key', '')
try:
    if not default_key and "GEMINI_API_KEY" in st.secrets:
        default_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

input_key = st.sidebar.text_input(
    "API Key:",
    value=default_key,
    type="password",
    help="Le chiavi create su AI Studio nel 2026 iniziano con 'AQ.'. Le vecchie 'AIza' sono state dismesse da Google a settembre 2026."
)

if input_key and input_key.strip() != st.session_state['saved_api_key']:
    st.session_state['saved_api_key'] = input_key.strip()
    st.session_state['key_validated'] = False
    st.session_state['available_models'] = []
    st.session_state['autocheck_done'] = False

active_api_key = st.session_state.get('saved_api_key', '')

if active_api_key.startswith("AQ."):
    key_type = "Auth key — formato attuale Google 2026 ✔"
elif active_api_key.startswith("AIza"):
    key_type = "Standard key — formato legacy, dismesso da Google"
elif active_api_key:
    key_type = "Formato non riconosciuto"
else:
    key_type = None

# Verifica automatica al primo caricamento della chiave
if active_api_key and not st.session_state['autocheck_done']:
    with st.spinner("Verifica automatica della chiave…"):
        ok, msg, models = validate_key_live(active_api_key)
    st.session_state['key_validated'] = ok
    st.session_state['key_validation_msg'] = msg
    st.session_state['available_models'] = models
    st.session_state['autocheck_done'] = True

if active_api_key:
    st.sidebar.caption(f"Tipo: {key_type}")
    if st.sidebar.button("🔄 Riverifica connessione", use_container_width=True):
        with st.spinner("Verifica in corso…"):
            ok, msg, models = validate_key_live(active_api_key)
        st.session_state['key_validated'] = ok
        st.session_state['key_validation_msg'] = msg
        st.session_state['available_models'] = models

    if st.session_state['key_validated']:
        st.sidebar.success(f"🟢 {st.session_state['key_validation_msg']}")
    elif st.session_state['key_validation_msg']:
        st.sidebar.error(f"🔴 {st.session_state['key_validation_msg']}")
else:
    st.sidebar.warning("🔴 Nessuna API Key inserita.")
    st.sidebar.markdown("👉 [Genera la chiave su AI Studio](https://aistudio.google.com/apikey)")

st.sidebar.divider()

# ---------------------------------------------------------------------------
# SIDEBAR - PARAMETRI
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Parametri di Analisi")

detected = st.session_state.get('available_models', [])
model_options = detected if detected else FALLBACK_MODELS

if detected:
    st.sidebar.caption(f"✅ Lista caricata dal tuo account ({len(detected)} modelli)")
else:
    st.sidebar.caption("⚠️ Lista di ripiego — verifica la chiave per caricare i modelli reali")

model_choice = st.sidebar.selectbox("Modello Gemini:", model_options, index=0)

fps_choice = st.sidebar.select_slider(
    "Campionamento (fps):",
    options=[0.2, 0.5, 1.0, 2.0],
    value=1.0,
    help="1.0 = un frame al secondo dell'intero video."
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
    m = re.search(r'(?:youtu\.be/|youtube\.com/(?:embed/|v/|watch\?v=|watch\?.+&v=|shorts/))([\w-]{11})', clean)
    if m:
        return m.group(1)
    if len(clean) == 11 and '/' not in clean:
        return clean
    return None


def build_prompt(active_targets, fps):
    wanted = [k for k, v in active_targets.items() if v] or ["qualsiasi materiale da imballaggio"]
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
- PELASPAN: chip o trucioli sfusi a forma di S, bianchi o verdi.
- CARTONE: scatole in cartone ondulato, nastro adesivo da imballaggio, etichette.

REGOLE OBBLIGATORIE:
1. Fornisci il timestamp REALE osservato nel video, formato MM:SS.
2. Dai priorità alle AZIONI (operatore o macchina che inserisce, avvolge o eroga
   il materiale) rispetto alla semplice presenza passiva sullo sfondo.
3. Non inventare timestamp. Se un materiale non compare, non elencarlo.
4. Se lo stesso materiale resta visibile a lungo, indica una sola riga con il
   momento più rappresentativo.
5. La confidenza deve riflettere quanto sei realmente sicuro.

Campionamento richiesto: circa {fps} frame al secondo.

Rispondi ESCLUSIVAMENTE con un array JSON valido, nessun testo prima o dopo:
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


def analyze_youtube_native(api_key, youtube_url, model_name, prompt, fps, fallback_models):
    """
    Passa l'URL YouTube direttamente a Gemini: il video è elaborato sui server
    Google, senza download locale. Se il modello scelto non è disponibile,
    ripiega automaticamente sugli altri modelli noti.
    """
    diag = {
        "sdk": "google-genai",
        "model_requested": model_name,
        "model_used": None,
        "method": None,
        "attempts": [],
        "raw_response_excerpt": None,
        "error": None,
    }

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        diag["error"] = f"SDK google-genai non disponibile: {e}"
        return None, diag

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        diag["error"] = f"Client non inizializzabile: {type(e).__name__}: {e}"
        return None, diag

    # Coda modelli: quello scelto, poi i fallback non ancora provati
    model_queue = [model_name] + [m for m in fallback_models if m != model_name]

    for candidate in model_queue:
        strategies = [
            ("con_fps", lambda: types.Part(
                file_data=types.FileData(file_uri=youtube_url),
                video_metadata=types.VideoMetadata(fps=fps),
            )),
            ("senza_fps", lambda: types.Part(
                file_data=types.FileData(file_uri=youtube_url)
            )),
        ]

        model_unavailable = False

        for label, part_builder in strategies:
            try:
                video_part = part_builder()
                response = client.models.generate_content(
                    model=candidate,
                    contents=types.Content(parts=[video_part, types.Part(text=prompt)]),
                )
                text = (response.text or "").strip()
                diag["attempts"].append({
                    "model": candidate, "strategy": label, "outcome": "risposta ricevuta"
                })
                diag["raw_response_excerpt"] = text[:800]

                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    for item in parsed:
                        item["engine_used"] = f"Gemini nativo su URL YouTube ({candidate}, {label})"
                    diag["method"] = label
                    diag["model_used"] = candidate
                    return parsed, diag

                diag["attempts"][-1]["outcome"] = "risposta senza array JSON"
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                diag["attempts"].append({
                    "model": candidate, "strategy": label, "outcome": err
                })
                # Se il modello non esiste, inutile provare l'altra strategia
                if "404" in err or "NOT_FOUND" in err or "no longer available" in err:
                    model_unavailable = True
                    break

        if model_unavailable:
            continue

    diag["error"] = "Nessun modello disponibile ha prodotto un risultato. Vedi 'attempts'."
    return None, diag


# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Video da analizzare")
    video_url = st.text_input(
        "URL YouTube:",
        value="https://www.youtube.com/watch?v=Uo3pD0sRbII",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    btn = st.button(
        "🚀 Analizza video con Gemini",
        type="primary",
        disabled=not bool(active_api_key),
        use_container_width=True,
    )
    if not active_api_key:
        st.caption("⚠️ Inserisci una API Key nella barra laterale per abilitare l'analisi.")

    video_id = extract_youtube_id(video_url)
    if video_id:
        st.video(f"https://www.youtube.com/watch?v={video_id}")
    elif video_url:
        st.warning("URL YouTube non riconosciuto.")

with col_right:
    st.subheader("2. Materiali rilevati")

    if btn and video_id and active_api_key:
        clean_url = f"https://www.youtube.com/watch?v={video_id}"
        prompt = build_prompt(targets, fps_choice)

        with st.spinner(f"Gemini sta guardando l'intero video a {fps_choice} fps… (30-90 s)"):
            results, diag = analyze_youtube_native(
                active_api_key, clean_url, model_choice, prompt, fps_choice,
                model_options
            )

        if results is None:
            st.error("❌ Analisi non riuscita. Nessun risultato inventato — ecco il motivo tecnico:")
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
                "model": diag.get("model_used") or model_choice,
                "fps": fps_choice,
                "analyzed_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
            st.session_state['current_analysis'] = analysis_obj
            st.session_state['history'] = [
                h for h in st.session_state['history'] if h['video_id'] != video_id
            ] + [analysis_obj]

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
            use_container_width=True,
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
