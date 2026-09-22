import streamlit as st
import re
import json
import time
import threading
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

CATEGORY_ICONS = {
    "PAPERplus": "🟤",
    "AIRplus": "🎈",
    "FOAMplus": "🔲",
    "PELASPAN": "⚪",
    "CARTONE": "📦",
}

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

st.sidebar.subheader("⏱️ Robustezza")
retry_on_overload = st.sidebar.checkbox(
    "Riprova automaticamente se il modello è sovraccarico (503)",
    value=True
)
timeout_seconds = st.sidebar.slider(
    "Timeout massimo complessivo (secondi)",
    min_value=60, max_value=600, value=240, step=30,
    help="Oltre questo tempo l'analisi si interrompe con un errore leggibile invece di restare appesa."
)

st.sidebar.subheader("📋 Visualizzazione risultati")
view_mode = st.sidebar.radio(
    "Formato:",
    ["Elenco compatto", "Tabella", "Schede dettagliate"],
    index=0,
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
        elapsed = h.get('elapsed_seconds')
        suffix = f" · {elapsed}s" if elapsed else ""
        if st.sidebar.button(f"▶ {h['video_id']} ({len(h['results'])} esiti{suffix})", key=f"hist_{idx}"):
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
- Il campo "title" deve essere BREVE: massimo 6 parole, senza punto finale.
- Nel campo "position" indica dove si trova nell'inquadratura (es. "in basso a sinistra",
  "centro", "sfondo destro", "primo piano").

Campionamento richiesto: circa {fps} frame al secondo.

Rispondi ESCLUSIVAMENTE con un array JSON valido, nessun testo prima o dopo, nessun blocco markdown:
[
  {{
    "time": "00:21",
    "seconds": 21,
    "title": "Fascio di carta Kraft su banco",
    "description": "Un ammasso di carta Kraft marrone accartocciata è appoggiato sul tavolo, parzialmente tagliato dal bordo inferiore sinistro dell'inquadratura.",
    "confidence": "72%",
    "category": "PAPERplus",
    "position": "in basso a sinistra",
    "action": false
  }}
]
Se non rilevi alcun materiale da imballaggio, rispondi con: []
"""


def run_analysis_worker(api_key, youtube_url, model_name, prompt, fps,
                        fallback_models, retry_503, shared):
    """
    Esegue l'analisi in un thread separato, aggiornando 'shared' passo per passo
    cosi che l'interfaccia possa mostrare lo stato in tempo reale.
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
    shared["diag"] = diag

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        diag["error"] = f"SDK google-genai non disponibile: {e}"
        shared["done"] = True
        return

    try:
        shared["status"] = "Inizializzazione del client Gemini…"
        client = genai.Client(api_key=api_key)
    except Exception as e:
        diag["error"] = f"Client non inizializzabile: {type(e).__name__}: {e}"
        shared["done"] = True
        return

    model_queue = [model_name] + [m for m in fallback_models if m != model_name]

    def try_call(candidate, with_fps):
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
                shared["status"] = (
                    f"Modello {candidate} · strategia {label} · tentativo {attempt_n + 1}/{max_tries} — "
                    f"Gemini sta elaborando il video…"
                )
                t_call = time.time()
                try:
                    text = try_call(candidate, with_fps)
                    call_s = round(time.time() - t_call, 1)
                    diag["attempts"].append({
                        "model": candidate, "strategy": label,
                        "try": attempt_n + 1, "seconds": call_s,
                        "outcome": "risposta ricevuta"
                    })
                    diag["raw_response_excerpt"] = text[:1000]
                    shared["status"] = f"Risposta ricevuta da {candidate} in {call_s}s · parsing…"

                    cleaned = re.sub(r'^```(?:json)?|```$', '', text, flags=re.MULTILINE).strip()
                    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                        for item in parsed:
                            item["engine_used"] = f"Gemini nativo su URL YouTube ({candidate}, {label})"
                        diag["method"] = label
                        diag["model_used"] = candidate
                        shared["results"] = parsed
                        shared["done"] = True
                        return

                    diag["attempts"][-1]["outcome"] = "risposta senza array JSON"
                    break

                except Exception as e:
                    call_s = round(time.time() - t_call, 1)
                    err = f"{type(e).__name__}: {e}"
                    diag["attempts"].append({
                        "model": candidate, "strategy": label,
                        "try": attempt_n + 1, "seconds": call_s, "outcome": err
                    })

                    if "404" in err or "NOT_FOUND" in err or "no longer available" in err:
                        shared["status"] = f"{candidate} non disponibile · passo al modello successivo"
                        model_dead = True
                        break

                    if ("503" in err or "UNAVAILABLE" in err) and attempt_n < max_tries - 1:
                        wait = 3 * (attempt_n + 1)
                        shared["status"] = f"{candidate} sovraccarico (503) · attendo {wait}s e riprovo…"
                        time.sleep(wait)
                        continue

                    break

            if model_dead:
                break

        if model_dead:
            continue

    diag["error"] = "Nessun modello disponibile ha prodotto un risultato. Vedi 'attempts'."
    shared["done"] = True


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

        shared = {"status": "Avvio…", "results": None, "diag": None, "done": False}

        worker = threading.Thread(
            target=run_analysis_worker,
            args=(active_api_key, clean_url, model_choice, prompt, fps_choice,
                  model_options, retry_on_overload, shared),
            daemon=True,
        )

        clock_box = st.empty()
        status_box = st.empty()
        bar = st.progress(0)

        t_start = time.time()
        worker.start()

        timed_out = False
        while not shared["done"]:
            elapsed = time.time() - t_start
            if elapsed > timeout_seconds:
                timed_out = True
                break

            pct = min(int(elapsed / timeout_seconds * 100), 99)
            bar.progress(pct)
            clock_box.markdown(
                f"**⏱️ {int(elapsed)}s** trascorsi / limite {timeout_seconds}s"
            )
            status_box.caption(f"🔄 {shared['status']}")
            time.sleep(1)

        total_elapsed = round(time.time() - t_start, 1)
        bar.empty()
        clock_box.empty()
        status_box.empty()

        diag = shared.get("diag") or {}
        diag["elapsed_seconds"] = total_elapsed
        diag["timeout_limit_seconds"] = timeout_seconds

        if timed_out:
            diag["error"] = (
                f"TIMEOUT dopo {total_elapsed}s. L'analisi è stata interrotta dall'app "
                f"per evitare il blocco della sessione. Ultimo stato: {shared['status']}"
            )
            st.error(
                f"⏱️ **Timeout dopo {total_elapsed} secondi.** "
                f"Ultimo stato: *{shared['status']}*"
            )
            with st.expander("🔧 Diagnostica tecnica", expanded=True):
                st.json(diag)

        elif shared["results"] is None:
            st.error(f"❌ Analisi fallita dopo {total_elapsed}s.")
            with st.expander("🔧 Diagnostica tecnica", expanded=True):
                st.json(diag)

        elif len(shared["results"]) == 0:
            st.warning(
                f"Nessun materiale da imballaggio rilevato (analisi completata in {total_elapsed}s)."
            )
            with st.expander("🔧 Diagnostica tecnica"):
                st.json(diag)

        else:
            analysis_obj = {
                "video_id": video_id,
                "youtube_url": clean_url,
                "results": shared["results"],
                "diagnostics": diag,
                "model": diag.get("model_used") or model_choice,
                "fps": fps_choice,
                "mode": mode_label,
                "min_confidence": min_confidence,
                "elapsed_seconds": total_elapsed,
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
        results = sorted(current['results'], key=lambda r: r.get('seconds', 0))
        vid = current['video_id']

        n_action = sum(1 for r in results if r.get("action"))
        elapsed = current.get('elapsed_seconds')
        cats = sorted({r.get('category', '?') for r in results})

        st.caption(
            f"**{len(results)} rilevamenti** · {n_action} azioni · "
            f"{', '.join(cats)} · `{current.get('model')}` · {elapsed}s"
        )

        # ---- Elenco compatto (default) ----
        if view_mode == "Elenco compatto":
            lines = []
            for r in results:
                icon = CATEGORY_ICONS.get(r.get("category", ""), "•")
                marker = "🎬" if r.get("action") else ""
                t = r.get("time", "??:??")
                secs = r.get("seconds", 0)
                conf = str(r.get("confidence", "")).replace("%", "")
                title = r.get("title", "Materiale")
                pos = r.get("position", "")
                pos_txt = f" · _{pos}_" if pos else ""
                link = f"https://www.youtube.com/watch?v={vid}&t={secs}s"
                lines.append(
                    f"- {icon}{marker} **[{t}]({link})** — {title} "
                    f"`{conf}%`{pos_txt}"
                )
            st.markdown("\n".join(lines))

            with st.expander("Descrizioni estese"):
                for r in results:
                    st.markdown(
                        f"**{r.get('time')}** — {r.get('title')}  \n"
                        f"<span style='font-size:13px;opacity:0.8'>{r.get('description','')}</span>",
                        unsafe_allow_html=True,
                    )

        # ---- Tabella ----
        elif view_mode == "Tabella":
            rows = [{
                "Tempo": r.get("time", ""),
                "Categoria": r.get("category", ""),
                "Materiale": r.get("title", ""),
                "Conf.": r.get("confidence", ""),
                "Posizione": r.get("position", ""),
                "Azione": "Sì" if r.get("action") else "",
            } for r in results]
            st.dataframe(rows, use_container_width=True, hide_index=True)

            st.caption("Link diretti ai momenti:")
            links = " · ".join(
                f"[{r.get('time')}](https://www.youtube.com/watch?v={vid}&t={r.get('seconds',0)}s)"
                for r in results
            )
            st.markdown(links)

        # ---- Schede dettagliate ----
        else:
            for r in results:
                icon = CATEGORY_ICONS.get(r.get("category", ""), "•")
                marker = " 🎬" if r.get("action") else ""
                with st.container(border=True):
                    st.markdown(
                        f"**{icon} {r.get('time','??:??')} — {r.get('title','Materiale')}**{marker} "
                        f"`{r.get('confidence','n/d')}`"
                    )
                    st.caption(r.get("description", ""))
                    meta = []
                    if r.get("category"):
                        meta.append(r["category"])
                    if r.get("position"):
                        meta.append(r["position"])
                    secs = r.get("seconds", 0)
                    meta.append(f"[▶ vai al {r.get('time')}](https://www.youtube.com/watch?v={vid}&t={secs}s)")
                    st.caption(" · ".join(meta))

        st.divider()

        audit = {
            "storopack_vision_ai_audit": {
                "generated_at": current.get("analyzed_at_utc"),
                "elapsed_seconds": current.get("elapsed_seconds"),
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

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "📥 Report JSON",
            data=json.dumps(audit, indent=2, ensure_ascii=False),
            file_name=f"storopack_audit_{vid}.json",
            mime="application/json",
            use_container_width=True,
        )
        csv_lines = ["tempo,secondi,categoria,materiale,confidenza,posizione,azione"]
        for r in results:
            csv_lines.append(
                f"\"{r.get('time','')}\",{r.get('seconds',0)},"
                f"\"{r.get('category','')}\",\"{r.get('title','')}\","
                f"\"{r.get('confidence','')}\",\"{r.get('position','')}\","
                f"{'si' if r.get('action') else 'no'}"
            )
        dl2.download_button(
            "📊 Report CSV",
            data="\n".join(csv_lines),
            file_name=f"storopack_audit_{vid}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("🔧 Diagnostica tecnica"):
            st.json(current.get("diagnostics", {}))
    elif not btn:
        st.info("Inserisci un URL YouTube e avvia l'analisi.")
