import os
import tempfile
from typing import Optional

import streamlit as st
import pandas as pd

from core.engine import PercussionOrchestratorEngine

st.set_page_config(
    page_title="Percussion Orchestrator",
    layout="wide",
)

left, mid, right = st.columns([1, 2, 1])
with mid:
    st.image("perc_orchestrator_web/assets/PO_logo.png", width="stretch")

# ------------------ Config dataset ------------------
st.subheader("1- Dataset")

colA, colB = st.columns([2, 3])

with colA:
    project_root = st.text_input("Carpeta del dataset", value="perc_dataset")

with colB:
    default_zip = "https://drive.google.com/file/d/12l3uRo7kNFd_DMuYHV2ZE2c0_LsA-2EJ/view?usp=drive_link"
    zip_url = st.text_input("ZIP público (Drive) si falta el dataset (opcional)", value=default_zip)

@st.cache_resource(show_spinner=True)
def get_engine(project_root: str, zip_url: Optional[str]):
    # zip_url puede ser "" → None
    z = zip_url.strip() if zip_url else ""
    z = z if z else None
    return PercussionOrchestratorEngine(project_root=project_root, zip_url=z)

try:
    engine = get_engine(project_root, zip_url)
    st.success("✅ Dataset listo")
    st.caption(f"Audio dir: {engine.paths.AUDIO_DIR}")
    st.caption(f"CSV: {engine.paths.CSV_PATH} (sep: {engine.csv_sep})")
except Exception as e:
    st.error("No se pudo preparar el dataset.")
    st.code(str(e))
    st.stop()

# ------------------ Entrada ------------------
st.markdown("---")
st.subheader("2- Audio de entrada")

uploaded = st.file_uploader("Sube un audio (wav/mp3/flac/ogg)", type=["wav", "mp3", "flac", "ogg"])

test_files = engine.list_test_sounds()
use_test = st.checkbox("Usar un test sound del dataset (en vez de subir archivo)", value=False)

picked_test = None
if use_test:
    if not test_files:
        st.warning("No hay carpeta test_sounds/ o está vacía.")
    else:
        picked_test = st.selectbox("Test sound", test_files)

# Resolver input_path
input_path = None
input_label = None

if use_test and picked_test:
    input_path = os.path.join(engine.paths.TEST_DIR, picked_test)
    input_label = picked_test
elif uploaded is not None:
    suffix = "." + uploaded.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        input_path = tmp.name
    input_label = uploaded.name

if input_path is None:
    st.info("Sube un audio o elige un test sound.")
    st.stop()

# ------------------ Parámetros ------------------
st.markdown("---")
st.subheader("3- Parámetros")

METHOD_LABEL = {
    "weighted": "Solo por sonido (rápido)",
    "hybrid": "Sonido + etiquetas (equilibrado)",
    "hybrid_quota": "Sonido + etiquetas con cuotas (más control)",
    "combined": "Mezcla total (sonido + etiquetas)",
}

OFFSETS_LABEL = {
    "envelope": "Por envolvente (más estable)",
    "peak": "Por pico (más directo)",
}

LAYER_LABEL = {
    "attack_layer": "Ataque",
    "body_layer": "Cuerpo",
    "resonance_layer": "Resonancia",
    "noise_layer": "Ruido",
    "special_fx": "FX / Especial",
}

col1, col2, col3 = st.columns(3)

with col1:
    method = st.selectbox(
        "Modo de selección",
        ["weighted", "hybrid", "hybrid_quota", "combined"],
        index=2,
        format_func=lambda k: METHOD_LABEL[k],
        help="Define si la app decide solo por el sonido o también usando información del dataset (material, registro, rol, etc.).",
    )

    n_return = st.slider(
        "Opciones a combinar",
        5, 60, 20, 1,
        help="Cuántos elementos mezcla para formar el resultado. Más alto = más mezcla; más bajo = más directo.",
    )

    K = st.slider(
        "Variedad (candidatos que explora)",
        30, 400, 200, 10,
        help="Cuanto más alto, más alternativas explora antes de decidir (más diverso, algo más lento).",
    )

with col2:
    # Solo tiene sentido en modos hybrid/hybrid_quota
    if method in ("hybrid", "hybrid_quota"):
        lambda_meta = st.slider(
            "Parecido al audio (perfil)",
            0.0, 2.0, 0.8, 0.05,
            help="Sube esto si quieres que el resultado se parezca más al 'tipo' de audio que has subido.",
        )
    else:
        lambda_meta = 0.0

    # Solo tiene sentido en modo combined
    if method == "combined":
        beta = st.slider(
            "Peso de las etiquetas del dataset",
            0.0, 2.0, 0.6, 0.05,
            help="Sube esto si quieres que la info del dataset (material, registro, rol…) mande más que el sonido.",
        )
    else:
        beta = 0.0

    do_crop = st.checkbox(
        "Recortar al largo del audio",
        True,
        help="Si está activado, la salida se ajusta a la duración del audio que has subido.",
    )

with col3:
    use_offsets = st.checkbox(
        "Alinear automáticamente",
        True,
        help="Intenta sincronizar el análisis con el audio para comparar mejor.",
    )

    offsets_mode = st.selectbox(
        "Cómo alinear",
        ["envelope", "peak"],
        index=0,
        format_func=lambda k: OFFSETS_LABEL[k],
        disabled=not use_offsets,
        help="Envolvente = más estable. Pico = se fija en el golpe más fuerte.",
    )

    env_dur = st.slider(
        "Ventana de análisis (s)",
        0.5, 8.0, 3.0, 0.5,
        disabled=not use_offsets,
        help="Cuánto audio usa para comparar. Más alto = más contexto.",
    )

    max_shift = st.slider(
        "Desplazamiento máximo (s)",
        0.0, 3.0, 0.8, 0.1,
        disabled=not use_offsets,
        help="Hasta cuánto puede 'mover' el audio internamente para alinearlo.",
    )

# --- Cuotas (solo hybrid_quota) ---
if method == "hybrid_quota":
    with st.expander("Cuotas por capa (solo en modo con cuotas)", expanded=False):
        q_attack = st.slider(LAYER_LABEL["attack_layer"], 0, 6, 2, 1)
        q_body   = st.slider(LAYER_LABEL["body_layer"], 0, 6, 2, 1)
        q_res    = st.slider(LAYER_LABEL["resonance_layer"], 0, 6, 2, 1)
        q_noise  = st.slider(LAYER_LABEL["noise_layer"], 0, 6, 1, 1)
        q_fx     = st.slider(LAYER_LABEL["special_fx"], 0, 6, 1, 1)
else:
    # Valores por defecto si no aplica
    q_attack, q_body, q_res, q_noise, q_fx = 2, 2, 2, 1, 1

# --- Ganancias ---
with st.expander("Ganancias por capa", expanded=False):
    g_attack = st.slider(f"{LAYER_LABEL['attack_layer']} (ganancia)", 0.0, 2.0, 1.0, 0.05)
    g_body   = st.slider(f"{LAYER_LABEL['body_layer']} (ganancia)",   0.0, 2.0, 0.7, 0.05)
    g_res    = st.slider(f"{LAYER_LABEL['resonance_layer']} (ganancia)", 0.0, 2.0, 0.6, 0.05)
    g_noise  = st.slider(f"{LAYER_LABEL['noise_layer']} (ganancia)",  0.0, 2.0, 0.6, 0.05)
    g_fx     = st.slider(f"{LAYER_LABEL['special_fx']} (ganancia)",   0.0, 2.0, 0.8, 0.05)

quotas = {
    "attack_layer": q_attack, "body_layer": q_body, "resonance_layer": q_res,
    "noise_layer": q_noise, "special_fx": q_fx
}
gain_map = {
    "attack_layer": g_attack, "body_layer": g_body, "resonance_layer": g_res,
    "noise_layer": g_noise, "special_fx": g_fx
}

# ------------------ Run + Historial ------------------
st.markdown("---")
st.subheader("4- Procesar")

if "history" not in st.session_state:
    st.session_state.history = []

run = st.button("▶ Ejecutar", type="primary")

if run:
    with st.spinner("Procesando..."):
        out_path, result, profile = engine.run_orchestrator(
            input_audio_path=input_path,
            method=method,
            n_return=n_return,
            K=K,
            lambda_meta=lambda_meta,
            beta=beta,
            use_offsets=use_offsets,
            offsets_mode=offsets_mode,
            env_dur=env_dur,
            max_shift=max_shift,
            do_crop=do_crop,
            quotas=quotas,
            gain_map=gain_map,
        )

        # Guardar bytes para que el historial no dependa del tmpfile
        with open(input_path, "rb") as f:
            in_bytes = f.read()
        with open(out_path, "rb") as f:
            out_bytes = f.read()

        st.session_state.history.append({
            "label": input_label,
            "method": method,
            "out_path": out_path,
            "in_bytes": in_bytes,
            "out_bytes": out_bytes,
            "result": result,
            "profile": profile,
        })

# Mostrar historial (último por defecto)
if st.session_state.history:
    st.subheader("Resultados (última ejecución)")
    item = st.session_state.history[-1]

    a, b = st.columns(2)
    with a:
        st.caption("Original")
        st.audio(item["in_bytes"])
    with b:
        st.caption("Generado")
        st.audio(item["out_bytes"])

    if item["profile"] is not None:
        st.info(f"Perfil inferido: {item['profile']}")

    df_show: pd.DataFrame = item["result"]
    show_cols = [c for c in ["file","family","orchestration_role","score","meta_penalty","final_score","distance","time_offset_s","env_corr"] if c in df_show.columns]
    st.dataframe(df_show[show_cols] if show_cols else df_show, use_container_width=True)

    st.download_button(
        "⬇️ Descargar WAV generado",
        data=item["out_bytes"],
        file_name=os.path.basename(item["out_path"]),
        mime="audio/wav"
    )

    with st.expander("Historial (ejecuciones anteriores)", expanded=False):
        for i, it in enumerate(reversed(st.session_state.history), start=1):
            st.write(f"{i}. {it['label']} — {it['method']} — {os.path.basename(it['out_path'])}")

    clear = st.button("🧹 Borrar resultados (vaciar historial)")
    if clear:
        st.session_state.history = []
        st.experimental_rerun()
