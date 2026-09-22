import streamlit as st
import re
import json
import time
import threading
from datetime import datetime, date

st.set_page_config(
    page_title="Storopack Vision AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# STILE
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
  h1, h2, h3 { letter-spacing: -0.01em; }
  div[data-testid="stMetricValue"] { font-size: 1.4rem; }
  .sp-hero {
      background: linear-gradient(120deg, #002b55 0%, #0061c9 100%);
      padding: 26px 30px; border-radius: 14px; color: #fff; margin-bottom: 22px;
  }
  .sp-hero h1 { margin: 0; font-size: 25px; font-weight: 700; }
  .sp-hero p  { margin: 6px 0 0 0; font-size: 14px; opacity: .88; }
  .sp-row {
      display: flex; align-items: center; gap: 12px;
      padding: 9px 12px; border-radius: 9px; margin-bottom: 6px;
      background: rgba(128,128,128,.06);
      border-left: 3px solid var(--sp-accent, #0061c9);
  }
  .sp-row:hover { background: rgba(0,97,201,.10); }
  .sp-time {
      font-family: ui-monospace, SFMono-Regular, monospace;
      font-weight: 650; font-size: 13px; min-width: 52px;
      color: #0061c9; text-decoration: none;
  }
  .sp-title { flex: 1; font-size: 14px; }
  .sp-meta  { font-size: 12px; opacity: .62; white-space: nowrap; }
  .sp-badge {
      font-size: 11px; padding: 2px 8px; border-radius: 10px;
      background: rgba(0,97,201,.12); color: #0061c9; font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

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
    ('usage_counter', {}),
    ('quota_exhausted', {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

FALLBACK_MODELS = [
    "gemini-3.6-flash", "gemini-3.6-pro", "gemini-3.5-flash",
    "gemini-3.0-flash", "gemini-2.0-flash",
]

CAT = {
    "PAPERplus": ("🟤", "#8a6d3b"),
    "AIRplus":   ("🎈", "#2b8ac6"),
    "FOAMplus":  ("🔲", "#6b7280"),
    "PELASPAN":  ("⚪", "#4b9e6a"),
    "CARTONE":   ("📦", "#b07b3f"),
}

TODAY = date.today().isoformat()


def bump_usage(model):
    st.session_state['usage_counter'].setdefault(TODAY, {})
    st.session_state['usage_counter'][TODAY][model] = \
        st.session_state['usage_counter'][TODAY].get(model, 0) + 1


def usage_today(model):
    return st.session_state['usage_counter'].get(TODAY, {}).get(model, 0)


def mark_exhausted(model):
    st.session_state['quota_exhausted'][model] = TODAY


def is_exhausted(model):
    return st.session_state['quota_exhausted'].get(model) == TODAY


def validate_key_live(api_key):
    try:
        from google import genai
    except ImportError as e:
        return False, f"SDK google-genai non installato: {e}", []
    try:
        client = genai.Client(api_key=api_key)
        models = []
        for m in client.models.list():
            short = (getattr(m, "name", "") or "").replace("models/", "")
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
            return True, f"{len(models)} modelli disponibili", models
        return False, "Nessun modello generativo compatibile", []
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", []


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


def classify_error(err_text):
    if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text:
        m = re.search(r"limit:\s*(\d+)", err_text)
        return "quota", {"limit": m.group(1) if m else "?"}
    if "404" in err_text or "NOT_FOUND" in err_text or "no longer available" in err_text:
        return "model_dead", {}
    if "503" in err_text or "UNAVAILABLE" in err_text:
        return "overload", {}
    return "other", {}


# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="sp-hero">
  <h1>📦 Storopack Vision AI</h1>
  <p>Rileva i materiali da imballaggio nei video YouTube &mdash; analisi nativa con Google Gemini</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Impostazioni")

    with st.expander("🔑 API Key", expanded=not st.session_state['key_validated']):
        default_key = st.session_state.get('saved_api_key', '')
        try:
            if not default_key and "GEMINI_API_KEY" in st.secrets:
                default_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

        input_key = st.text_input(
            "Chiave Gemini", value=default_key, type="password",
            label_visibility="collapsed",
            placeholder="AQ.Ab8... oppure AIza...",
        )
        if input_key and input_key.strip() != st.session_state['saved_api_key']:
            st.session_state.update({
                'saved_api_key': input_key.strip(), 'key_validated': False,
                'available_models': [], 'autocheck_done': False,
            })

        active_api_key = st.session_state.get('saved_api_key', '')

        if active_api_key and not st.session_state['autocheck_done']:
            with st.spinner("Verifica…"):
                ok, msg, models = validate_key_live(active_api_key)
            st.session_state.update({
                'key_validated': ok, 'key_validation_msg': msg,
                'available_models': models, 'autocheck_done': True,
            })

        if not active_api_key:
            st.caption("Nessuna chiave inserita.")
            st.link_button("Genera su AI Studio", "https://aistudio.google.com/apikey",
                           use_container_width=True)
        elif st.session_state['key_validated']:
            st.success(st.session_state['key_validation_msg'], icon="🟢")
        else:
            st.error(st.session_state['key_validation_msg'], icon="🔴")

        if active_api_key and st.button("Riverifica", use_container_width=True):
            with st.spinner("Verifica…"):
                ok, msg, models = validate_key_live(active_api_key)
            st.session_state.update({
                'key_validated': ok, 'key_validation_msg': msg, 'available_models': models,
            })
            st.rerun()

    detected = st.session_state.get('available_models', [])
    model_options = detected if detected else FALLBACK_MODELS

    def model_label(m):
        if is_exhausted(m):
            return f"{m} — ⛔ quota finita"
        u = usage_today(m)
        return f"{m} — {u}/20" if u else m

    with st.expander("🤖 Modello e precisione", expanded=True):
        model_choice = st.selectbox("Modello", model_options, index=0,
                                    format_func=model_label)
        if is_exhausted(model_choice):
            st.error("Quota esaurita per oggi su questo modello.", icon="⛔")

        fps_choice = st.select_slider(
            "Campionamento", options=[0.2, 0.5, 1.0, 2.0], value=1.0,
            format_func=lambda v: f"{v} fps",
            help="Più alto = più preciso, più lento e più costoso.",
        )

        exhaustive = st.toggle(
            "Modalità esaustiva", value=True,
            help="Rileva anche il materiale fermo sullo sfondo o ai bordi dell'inquadratura. "
                 "Disattiva per cercare solo le azioni di imballaggio.",
        )

        min_confidence = st.slider("Confidenza minima", 30, 90, 45, 5,
                                   format="%d%%")

    with st.expander("🎯 Materiali"):
        c1, c2 = st.columns(2)
        targets = {
            "PAPERplus": c1.checkbox("🟤 Carta", value=True),
            "AIRplus":   c2.checkbox("🎈 Aria", value=True),
            "FOAMplus":  c1.checkbox("🔲 Schiuma", value=True),
            "PELASPAN":  c2.checkbox("⚪ Chip", value=True),
            "CARTONE":   c1.checkbox("📦 Cartone", value=True),
        }

    with st.expander("⏱️ Avanzate"):
        timeout_seconds = st.slider("Timeout (s)", 60, 900, 420, 30)
        retry_on_overload = st.checkbox("Riprova se sovraccarico (503)", value=True)
        auto_fallback = st.checkbox("Cambia modello se quota finita", value=True)
        view_mode = st.radio("Vista risultati",
                             ["Elenco", "Tabella", "Schede"], index=0,
                             horizontal=True)

    today_usage = st.session_state['usage_counter'].get(TODAY, {})
    if today_usage:
        st.markdown("###### 📊 Uso di oggi")
        for m, n in sorted(today_usage.items(), key=lambda kv: -kv[1]):
            st.progress(min(n / 20, 1.0), text=f"{m} · {n}/20")

    if st.session_state['history']:
        st.markdown("###### 📜 Cronologia")
        for idx, h in enumerate(st.session_state['history'][::-1]):
            el = h.get('elapsed_seconds')
            if st.button(
                f"{h['video_id']} · {len(h['results'])} esiti" + (f" · {el}s" if el else ""),
                key=f"hist_{idx}", use_container_width=True,
            ):
                st.session_state['current_analysis'] = h
                st.rerun()

active_api_key = st.session_state.get('saved_api_key', '')


# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------
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
- Se lo stesso materiale compare in momenti distinti e separati del video, riporta
  ciascuna occorrenza.
- Il campo "title" deve essere BREVE: massimo 6 parole, senza punto finale.
- Nel campo "position" indica dove si trova nell'inquadratura.

Campionamento richiesto: circa {fps} frame al secondo.

Rispondi ESCLUSIVAMENTE con un array JSON valido, nessun testo prima o dopo, nessun blocco markdown:
[
  {{
    "time": "00:21", "seconds": 21,
    "title": "Fascio di carta Kraft su banco",
    "description": "Un ammasso di carta Kraft marrone accartocciata è appoggiato sul tavolo, parzialmente tagliato dal bordo inferiore sinistro.",
    "confidence": "72%", "category": "PAPERplus",
    "position": "in basso a sinistra", "action": false
  }}
]
Se non rilevi alcun materiale da imballaggio, rispondi con: []
"""


def run_analysis_worker(api_key, youtube_url, model_name, prompt, fps,
                        fallback_models, retry_503, allow_fallback, shared):
    diag = {
        "sdk": "google-genai", "model_requested": model_name, "model_used": None,
        "method": None, "attempts": [], "models_quota_exhausted": [],
        "raw_response_excerpt": None, "error": None, "error_type": None,
    }
    shared["diag"] = diag

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        diag.update(error=f"SDK non disponibile: {e}", error_type="sdk")
        shared["done"] = True
        return

    try:
        shared["status"] = "Connessione a Gemini…"
        client = genai.Client(api_key=api_key)
    except Exception as e:
        diag.update(error=f"Client non inizializzabile: {e}", error_type="client")
        shared["done"] = True
        return

    model_queue = ([model_name] + [m for m in fallback_models if m != model_name]
                   if allow_fallback else [model_name])

    def try_call(candidate, with_fps):
        if with_fps:
            vp = types.Part(file_data=types.FileData(file_uri=youtube_url),
                            video_metadata=types.VideoMetadata(fps=fps))
        else:
            vp = types.Part(file_data=types.FileData(file_uri=youtube_url))
        r = client.models.generate_content(
            model=candidate,
            contents=types.Content(parts=[vp, types.Part(text=prompt)]),
        )
        return (r.text or "").strip()

    last_kind = None

    for candidate in model_queue:
        skip = False
        for label, with_fps in [("con_fps", True), ("senza_fps", False)]:
            tries = 3 if retry_503 else 1
            for n in range(tries):
                shared["status"] = f"Analisi con {candidate} · tentativo {n + 1}/{tries}"
                shared.setdefault("models_called", set()).add(candidate)
                t0 = time.time()
                try:
                    text = try_call(candidate, with_fps)
                    dt = round(time.time() - t0, 1)
                    diag["attempts"].append({"model": candidate, "strategy": label,
                                             "try": n + 1, "seconds": dt,
                                             "outcome": "risposta ricevuta"})
                    diag["raw_response_excerpt"] = text[:1000]
                    shared["status"] = f"Risposta in {dt}s · elaborazione…"

                    cleaned = re.sub(r'^```(?:json)?|```$', '', text, flags=re.MULTILINE).strip()
                    m = re.search(r'\[.*\]', cleaned, re.DOTALL)
                    if m:
                        parsed = json.loads(m.group(0))
                        for it in parsed:
                            it["engine_used"] = f"{candidate} ({label})"
                        diag.update(method=label, model_used=candidate)
                        shared["results"] = parsed
                        shared["done"] = True
                        return
                    diag["attempts"][-1]["outcome"] = "risposta senza JSON"
                    break

                except Exception as e:
                    dt = round(time.time() - t0, 1)
                    err = f"{type(e).__name__}: {e}"
                    kind, info = classify_error(err)
                    last_kind = kind
                    diag["attempts"].append({"model": candidate, "strategy": label,
                                             "try": n + 1, "seconds": dt,
                                             "error_type": kind, "outcome": err[:400]})
                    if kind == "quota":
                        diag["models_quota_exhausted"].append(candidate)
                        shared.setdefault("exhausted", set()).add(candidate)
                        shared["status"] = f"⛔ {candidate}: quota finita"
                        skip = True
                        break
                    if kind == "model_dead":
                        shared["status"] = f"{candidate} non disponibile"
                        skip = True
                        break
                    if kind == "overload" and n < tries - 1:
                        w = 3 * (n + 1)
                        shared["status"] = f"{candidate} sovraccarico · riprovo tra {w}s"
                        time.sleep(w)
                        continue
                    break
            if skip:
                break
        if skip:
            continue

    if diag["models_quota_exhausted"]:
        diag.update(error_type="quota",
                    error="Quota esaurita: " + ", ".join(sorted(set(diag["models_quota_exhausted"]))))
    else:
        diag.update(error_type=last_kind or "unknown",
                    error="Nessun modello ha prodotto un risultato.")
    shared["done"] = True


# ---------------------------------------------------------------------------
# BARRA DI INPUT
# ---------------------------------------------------------------------------
in_col, btn_col = st.columns([4, 1])
video_url = in_col.text_input(
    "URL YouTube", value="https://www.youtube.com/watch?v=nG93C6N9Ky4",
    label_visibility="collapsed", placeholder="Incolla qui l'URL del video YouTube…",
)
btn_col.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
btn = btn_col.button("Analizza  ▶", type="primary", use_container_width=True,
                     disabled=not bool(active_api_key))

video_id = extract_youtube_id(video_url)

sub = []
sub.append(f"`{model_choice}`")
sub.append(f"{fps_choice} fps")
sub.append("esaustiva" if exhaustive else "solo azioni")
sub.append(f"≥{min_confidence}%")
st.caption(" · ".join(sub) + "  —  modifica in **⚙️ Impostazioni** (barra laterale)")

if not active_api_key:
    st.warning("Inserisci la API Key nella barra laterale per iniziare.", icon="🔑")
elif video_url and not video_id:
    st.error("URL YouTube non riconosciuto.", icon="⚠️")

st.divider()

# ---------------------------------------------------------------------------
# ESECUZIONE
# ---------------------------------------------------------------------------
if btn and video_id and active_api_key:
    clean_url = f"https://www.youtube.com/watch?v={video_id}"
    prompt = build_prompt(targets, fps_choice, exhaustive, min_confidence)
    shared = {"status": "Avvio…", "results": None, "diag": None, "done": False,
              "models_called": set(), "exhausted": set()}

    worker = threading.Thread(
        target=run_analysis_worker,
        args=(active_api_key, clean_url, model_choice, prompt, fps_choice,
              model_options, retry_on_overload, auto_fallback, shared),
        daemon=True,
    )

    prog_box = st.container()
    with prog_box:
        m1, m2 = st.columns([1, 4])
        clock = m1.empty()
        status = m2.empty()
        bar = st.progress(0)

    t0 = time.time()
    worker.start()
    timed_out = False

    while not shared["done"]:
        el = time.time() - t0
        if el > timeout_seconds:
            timed_out = True
            break
        bar.progress(min(int(el / timeout_seconds * 100), 99))
        clock.metric("Tempo", f"{int(el)}s")
        status.info(shared["status"], icon="⏳")
        time.sleep(1)

    total = round(time.time() - t0, 1)
    prog_box.empty()

    for m in shared.get("models_called", set()):
        bump_usage(m)
    for m in shared.get("exhausted", set()):
        mark_exhausted(m)

    diag = shared.get("diag") or {}
    diag["elapsed_seconds"] = total
    diag["timeout_limit_seconds"] = timeout_seconds

    if timed_out:
        diag.update(error_type="timeout", error=f"Timeout dopo {total}s")
        st.error(f"Timeout dopo {total}s — {shared['status']}", icon="⏱️")
        st.caption("Alza il timeout in **Impostazioni → Avanzate** oppure riduci gli fps.")
        with st.expander("Diagnostica"):
            st.json(diag)

    elif diag.get("error_type") == "quota":
        ex = sorted(set(diag.get("models_quota_exhausted", [])))
        st.error(f"Quota giornaliera esaurita — {', '.join(ex)}", icon="⛔")
        q1, q2, q3 = st.columns(3)
        q1.info("**Cambia modello**\n\nOgni modello ha quota separata.")
        q2.info("**Attendi il reset**\n\nMezzanotte Pacific · ~09:00 in Italia.")
        q3.info("**Attiva billing**\n\n~0,20 $ per video di 4 minuti.")
        st.link_button("Apri Google AI Studio", "https://aistudio.google.com/apikey")
        with st.expander("Diagnostica"):
            st.json(diag)

    elif shared["results"] is None:
        st.error(f"Analisi fallita dopo {total}s.", icon="❌")
        with st.expander("Diagnostica", expanded=True):
            st.json(diag)

    elif len(shared["results"]) == 0:
        st.warning(f"Nessun materiale rilevato · analisi completata in {total}s.", icon="🔍")
        st.caption("Prova ad abbassare la confidenza minima o ad aumentare gli fps.")
        with st.expander("Diagnostica"):
            st.json(diag)

    else:
        obj = {
            "video_id": video_id, "youtube_url": clean_url,
            "results": shared["results"], "diagnostics": diag,
            "model": diag.get("model_used") or model_choice,
            "fps": fps_choice, "mode": "esaustiva" if exhaustive else "solo azioni",
            "min_confidence": min_confidence, "elapsed_seconds": total,
            "analyzed_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        st.session_state['current_analysis'] = obj
        st.session_state['history'] = [
            h for h in st.session_state['history'] if h['video_id'] != video_id
        ] + [obj]
        st.rerun()

# ---------------------------------------------------------------------------
# RISULTATI
# ---------------------------------------------------------------------------
current = st.session_state.get('current_analysis')

if current and current.get('results'):
    results = sorted(current['results'], key=lambda r: r.get('seconds', 0))
    vid = current['video_id']
    n_action = sum(1 for r in results if r.get("action"))
    cats = sorted({r.get('category', '?') for r in results})

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rilevamenti", len(results))
    k2.metric("Azioni", n_action)
    k3.metric("Categorie", len(cats))
    k4.metric("Durata analisi", f"{current.get('elapsed_seconds', '?')}s")

    st.caption(f"Video `{vid}` · modello `{current.get('model')}` · "
               f"{current.get('fps')} fps · modalità {current.get('mode')}")

    st.divider()

    res_col, vid_col = st.columns([3, 2])

    with res_col:
        if view_mode == "Elenco":
            for r in results:
                icon, color = CAT.get(r.get("category", ""), ("•", "#0061c9"))
                secs = r.get("seconds", 0)
                conf = str(r.get("confidence", "")).replace("%", "")
                act = " 🎬" if r.get("action") else ""
                pos = f" · {r.get('position')}" if r.get("position") else ""
                st.markdown(f"""
<div class="sp-row" style="--sp-accent:{color}">
  <a class="sp-time" href="https://www.youtube.com/watch?v={vid}&t={secs}s" target="_blank">▶ {r.get('time','--:--')}</a>
  <span class="sp-title">{icon} {r.get('title','Materiale')}{act}</span>
  <span class="sp-meta">{conf}%{pos}</span>
</div>""", unsafe_allow_html=True)

            with st.expander("Descrizioni complete"):
                for r in results:
                    st.markdown(f"**{r.get('time')}** · {r.get('title')}")
                    st.caption(r.get("description", ""))

        elif view_mode == "Tabella":
            st.dataframe(
                [{
                    "⏱": r.get("time", ""), "Cat.": r.get("category", ""),
                    "Materiale": r.get("title", ""), "Conf.": r.get("confidence", ""),
                    "Posizione": r.get("position", ""),
                    "Azione": "✓" if r.get("action") else "",
                } for r in results],
                use_container_width=True, hide_index=True,
            )
            st.caption("Vai al momento: " + " · ".join(
                f"[{r.get('time')}](https://www.youtube.com/watch?v={vid}&t={r.get('seconds',0)}s)"
                for r in results))

        else:
            for r in results:
                icon, _ = CAT.get(r.get("category", ""), ("•", ""))
                with st.container(border=True):
                    a, b = st.columns([3, 1])
                    a.markdown(f"**{icon} {r.get('time')} — {r.get('title')}**"
                               + (" 🎬" if r.get("action") else ""))
                    b.markdown(f"<div style='text-align:right'><span class='sp-badge'>"
                               f"{r.get('confidence','n/d')}</span></div>",
                               unsafe_allow_html=True)
                    st.caption(r.get("description", ""))
                    st.caption(
                        f"{r.get('category','')} · {r.get('position','')} · "
                        f"[▶ vai](https://www.youtube.com/watch?v={vid}&t={r.get('seconds',0)}s)")

    with vid_col:
        st.video(f"https://www.youtube.com/watch?v={vid}")
        st.caption("Clicca un orario a sinistra per aprire il momento su YouTube.")

        audit = {"storopack_vision_ai_audit": {
            "generated_at": current.get("analyzed_at_utc"),
            "elapsed_seconds": current.get("elapsed_seconds"),
            "video_metadata": {
                "video_id": vid, "youtube_url": current.get("youtube_url"),
                "model": current.get("model"), "sampling_fps": current.get("fps"),
                "detection_mode": current.get("mode"),
                "min_confidence_pct": current.get("min_confidence"),
            },
            "api_diagnostics": current.get("diagnostics"),
            "detection_findings": results,
        }}
        csv = ["tempo,secondi,categoria,materiale,confidenza,posizione,azione"]
        for r in results:
            csv.append(f"\"{r.get('time','')}\",{r.get('seconds',0)},"
                       f"\"{r.get('category','')}\",\"{r.get('title','')}\","
                       f"\"{r.get('confidence','')}\",\"{r.get('position','')}\","
                       f"{'si' if r.get('action') else 'no'}")

        d1, d2 = st.columns(2)
        d1.download_button("JSON", json.dumps(audit, indent=2, ensure_ascii=False),
                           f"storopack_{vid}.json", "application/json",
                           use_container_width=True)
        d2.download_button("CSV", "\n".join(csv), f"storopack_{vid}.csv",
                           "text/csv", use_container_width=True)

        with st.expander("Diagnostica tecnica"):
            st.json(current.get("diagnostics", {}))

elif not btn:
    if video_id:
        p1, p2 = st.columns([3, 2])
        with p1:
            st.markdown("#### Pronto per l'analisi")
            st.caption("Premi **Analizza** per avviare la scansione del video. "
                       "Gemini esaminerà l'intero filmato e restituirà i timestamp "
                       "in cui compare materiale da imballaggio.")
            st.caption("Tempi indicativi: ~30s per un video di 1 minuto, "
                       "~2-4 min per un video di 4 minuti a 1 fps.")
        with p2:
            st.video(f"https://www.youtube.com/watch?v={video_id}")
    else:
        st.info("Incolla un URL YouTube per iniziare.", icon="👆")
