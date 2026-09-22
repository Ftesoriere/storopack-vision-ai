import streamlit as st
import re
import json
import time
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
            if any(x in short for x in ("embedding", "imagen", "veo", "tts", "aqa")):
                continue
            models.append(short)

        if models:
            models.sort(key=lambda s: ("3.6" not in s, "3.5" not in s, "flash" not in s, s))
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
    help="Le chiavi create su AI Studio nel 2026 iniziano con 'AQ.'."
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
    key_type = "Standard key — formato legacy dismesso"
elif active_api_key:
    key_type = "Formato non riconosciuto"
else:
    key_type = None

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
)

st.sidebar.subheader("🔍 Sensibilità di rilevamento")
detection_mode = st.sidebar.radio(
    "Modalità:",
    [
        "Esaustiva — ogni presenza, anche passiva o marginale",
        "Solo azioni — inserimento, avvolgimento, erogazione",
    ],
    index=0,
    help="La modalità esaustiva rileva anche materiale fermo sullo sfondo o ai bordi dell'inquadratura."
)
exhaustive = detection_mode.startswith("Esaustiva")

min_confidence = st.sidebar.slider(
    "Confidenza minima da riportare (%)",
    min_value=30, max_value=90, value=45, step=5,
    help="Abbassa il valore per far emergere anche i rilevamenti incerti."
)

retry_on_overload = st.sidebar.checkbox(
    "Riprova automaticamente se il modello è sovraccarico (503)",
    value=True
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


def build_prompt(active_targets, fps, exhaustive_mode, min_conf):
    wanted = [k for k, v in active_targets.items() if v] or ["qualsiasi materiale da imballaggio"]

    if exhaustive_mode:
        priority_block = """MODALITÀ ESAUSTIVA — REGOLE DI RILEVAMENTO:
1. Riporta OGNI presenza di materiale da imballaggio, indipendentemente dal fatto
   che qualcuno lo stia manipolando o meno. Materiale fermo su un tavolo, appoggiato
   su uno scaffale, impilato in un angolo o visibile solo parzialmente DEVE essere
   riportato allo stesso modo di quello manipolato attivamente.
2. Ispeziona sistematicamente TUTTA l'inquadratura, non solo il soggetto centrale:
   angoli, bordi, primo piano sfocato, sfondo, piani di lavoro laterali, oggetti
   tagliati dal bordo del fotogramma.
3. Includi anche materiale che occupa una porzione piccola dell'immagine (anche solo
   il 3-5% della superficie) purché sia riconoscibile.
4. Il campo "action" distingue i due casi: true se qualcuno lo sta manipolando,
   false se è semplicemente presente nella scena. Entrambi vanno riportati."""
    else:
        priority_block = """MODALITÀ AZIONI — REGOLE DI RILEVAMENTO:
1. Riporta i momenti in cui un operatore o una macchina inserisce, avvolge, eroga
   o manipola attivamente il materiale da imballaggio.
2. Ignora il materiale semplicemente presente sullo sfondo senza interazione."""

    return f"""Sei un ispettore tecnico specializzato in imballaggi protettivi industriali Storopack.
Il tuo compito è un censimento visivo rigoroso, non un riassunto del video.

Guarda l'INTERO video e individua il materiale da imballaggio protettivo.
Categorie da cercare: {', '.join(wanted)}.

RIFERIMENTI VISIVI DETTAGLIATI
- PAPERplus / PAPERwrap: carta Kraft marrone o bianca, stropicciata o accartocciata,
  fasci e strisce di carta, carta a fisarmonica, carta a nido d'ape, rotoli di carta,
  cuscini di carta, imbottitura di carta dentro o accanto alle scatole.
  ATTENZIONE: spesso appare come un ammasso informe di carta beige/marrone chiaro,
  facilmente scambiabile per stoffa o scarto. Verifica sempre questi ammassi.
- AIRplus / AIRmove: catene di cuscini d'aria in plastica trasparente, film a bolle
  d'aria (pluriball), sacchetti gonfiati, rotoli di film trasparente.
  ATTENZIONE: la plastica trasparente è poco visibile, cerca riflessi e superfici lucide.
- FOAMplus: schiuma poliuretanica espansa, sacchetti di schiuma auto-modellanti,
  inserti in polistirolo bianco sagomato.
- PELASPAN: chip o trucioli sfusi a forma di S, bianchi o verdi.
- CARTONE: scatole in cartone ondulato, nastro adesivo da imballaggio, etichette.

{priority_block}

REGOLE GENERALI
- Fornisci il timestamp REALE osservato nel video, formato MM:SS.
- Non inventare timestamp né materiali. Se una categoria non compare mai, non elencarla.
- Riporta un rilevamento anche se la tua confidenza è bassa, purché sia almeno {min_conf}%.
  Indica onestamente la confidenza reale: è preferibile un rilevamento incerto segnalato
  rispetto a un materiale presente ma taciuto.
- Se lo stesso materiale compare in momenti distinti e separati del video, riporta
  ciascuna occorrenza.
- Nel campo "position" indica dove si trova nell'inquadratura (es. "in basso a sinistra",
  "centro", "sfondo destro", "primo piano").

Campionamento richiesto: circa {fps} frame al secondo.

Rispondi ESCLUSIVAMENTE con un array JSON valido, nessun testo prima o dopo, nessun blocco markdown:
[
  {{
    "time": "00:21",
    "seconds": 21,
    "title": "Fascio di carta Kraft sul piano di lavoro",
    "description": "Un ammasso di carta Kraft marrone accartocciata è appoggiato sul tavolo, parzialmente tagliato dal bordo inferiore sinistro dell'inquadratura.",
    "confidence": "72%",
    "category": "PAPERplus",
    "position": "in basso a sinistra",
    "action": false
  }}
]
Se non rilevi alcun materiale da imballaggio, rispondi con: []
"""


def analyze_youtube_native(api_key, youtube_url, model_name, prompt, fps,
                           fallback_models, retry_503=True):
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

    model_queue = [model_name] + [m for m in fallback_models if m != model_name]

    def try_call(candidate, label, with_fps):
        if with_fps:
            video_part = types.Part(
                file_data=types.FileData(file_uri=youtube_url),
                video_metadata=types.VideoMetadata(fps=fps),
            )
        else:
            video_part = types.Part(file_data=types.FileData(file_uri=youtube_url))

        response = client.models.generate_content(
            model=candidate,
            contents=types.Content(parts=[video_part, types.Part(text=prompt)]),
        )
        return (response.text or "").strip()

    for candidate in model_queue:
        model_dead = False

        for label, with_fps in [("con_fps", True), ("senza_fps", False)]:
            max_tries = 3 if retry_503 else 1

            for attempt_n in range(max_tries):
                try:
                    text = try_call(candidate, label, with_fps)
                    diag["attempts"].append({
                        "model": candidate, "strategy": label,
                        "try": attempt_n + 1, "outcome": "risposta ricevuta"
                    })
                    diag["raw_response_excerpt"] = text[:1000]

                    cleaned = re.sub(r'^```(?:json)?|```$', '', text, flags=re.MULTILINE).strip()
                    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                        for item in parsed:
                            item["engine_used"] = f"Gemini nativo su URL YouTube ({candidate}, {label})"
                        diag["method"] = label
                        diag["model_used"] = candidate
                        return parsed, diag

                    diag["attempts"][-1]["outcome"] = "risposta senza array JSON"
                    break

                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    diag["attempts"].append({
                        "model": candidate, "strategy": label,
                        "try": attempt_n + 1, "outcome": err
                    })

                    if "404" in err or "NOT_FOUND" in err or "no longer available" in err:
                        model_dead = True
                        break

                    if ("503" in err or "UNAVAILABLE" in err) and attempt_n < max_tries - 1:
                        time.sleep(3 * (attempt_n + 1))
                        continue

                    break

            if model_dead:
                break

        if model_dead:
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
        value="https://www.youtube.com/watch?v=nG93C6N9Ky4",
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
        prompt = build_prompt(targets, fps_choice, exhaustive, min_confidence)

        mode_label = "esaustiva" if exhaustive else "solo azioni"
        with st.spinner(f"Gemini sta guardando l'intero video a {fps_choice} fps (modalità {mode_label})…"):
            results, diag = analyze_youtube_native(
                active_api_key, clean_url, model_choice, prompt, fps_choice,
                model_options, retry_on_overload
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
                "mode": mode_label,
                "min_confidence": min_confidence,
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

        n_action = sum(1 for r in results if r.get("action"))
        n_passive = len(results) - n_action

        st.success(
            f"✅ {len(results)} rilevamenti su `{vid}` "
            f"({n_action} azioni, {n_passive} presenze passive) — "
            f"modello `{current.get('model')}`, modalità `{current.get('mode', 'n/d')}`"
        )

        audit = {
            "storopack_vision_ai_audit": {
                "generated_at": current.get("analyzed_at_utc"),
                "video_metadata": {
                    "video_id": vid,
                    "youtube_url": current.get("youtube_url"),
                    "model": current.get("model"),
                    "sampling_fps": current.get("fps"),
                    "detection_mode": current.get("mode"),
                    "min_confidence_pct": current.get("min_confidence"),
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
            if item.get("position"):
                st.caption(f"📍 Posizione nell'inquadratura: {item['position']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidenza", item.get("confidence", "n/d"))
            c2.metric("Categoria", item.get("category", "n/d"))
            c3.metric("Tipo", "Azione" if item.get("action") else "Presenza")
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
