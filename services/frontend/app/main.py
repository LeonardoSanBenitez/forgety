import libs.env  # normalizes HF_TOKEN before any import below can read it; see libs/env.py

from typing import List, Dict, Optional, Any
import streamlit as st
import requests
import re
import traceback

from vision_unlearning.benchmarks.I_care import (
    domain_unlearning_algorithm,
    domain_task,
    domain_attribute,
    domain_entity,
    domain_model,
    domain_mp,
    domain_me,
    domain_s,
    domain_l,
    rt_name_to_class,
    rt_name_to_params,
    choose_metric_column_interference_per_entity,
)



def convert_mean_to_std(name: str) -> str:
    name = re.sub(r'\(.*?\)', '(~↓)', name)
    name = name.replace(' mean ', ' std ')
    return name


def metrics_to_markdown(metrics: List[Dict[str, float]]) -> str:
    name_to_value = {}
    for metric in metrics:
        name_to_value[metric['name']] = metric['value']

    output: str = ""
    for name in name_to_value:
        if '~' in name:
            continue
        if (' mean ' in name) and (convert_mean_to_std(name) in name_to_value):
            output += f"* **{name.replace(' mean', '')}**: {name_to_value[name]:.2f} ± {name_to_value[convert_mean_to_std(name)]:.1f}\n"
        elif (' std ' in name):
            continue
        else:
            output += f"* **{name}**: {name_to_value[name]:.2f}\n"
    return output

def request_rt_to_backend(template: str, model: str, task: str, params: Dict[str, Optional[str]]) -> None:
    try:
        with st.spinner("Sending request..."):
            response = requests.post(
                "http://backend:80/v1/public-api-compute-rt",
                json={
                    "template": template,
                    "params": {
                        "model": model,
                        "task": task,
                        **params,
                    },
                },
                timeout=180,
            )

        if response.status_code == 200:
            st.success("Request sent successfully!")
            fig, ax = rt_name_to_class[template].plot(response.json(), return_fig=True)
            st.pyplot(fig)
        else:
            st.error(f"Backend error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.code(traceback.format_exc())


def show_rt_graceful(template: str, model: str, task: str, params: Dict[str, Optional[str]]) -> None:
    """Compute and display an RT, showing a yellow warning on any failure instead of an error."""
    try:
        with st.spinner("Loading..."):
            response = requests.post(
                "http://backend:80/v1/public-api-compute-rt",
                json={
                    "template": template,
                    "params": {
                        "model": model,
                        "task": task,
                        **params,
                    },
                },
                timeout=180,
            )
        if response.status_code == 200:
            fig, ax = rt_name_to_class[template].plot(response.json(), return_fig=True)
            st.pyplot(fig)
        else:
            st.warning(f"Results not yet pre-computed (backend: {response.status_code}).")
    except Exception:
        st.warning("Results not yet pre-computed or backend unavailable.")

st.set_page_config(
    page_title="Forgety — Machine Unlearning as a Service",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------ CSS ------------------
st.markdown("""
<style>

:root {
--primary: #002668;
//--primary-dark: #3E885B;
--gradient-start: #002668;
--gradient-end: #2154ac;
//--bg: #C0D7BB;
//--card: #BEDCFE;
--text: white;
--muted: white;
}
            


html, body, [data-testid="stAppViewContainer"] {
background-color: var(--bg);
color: var(--text);
}

section {
padding: 5rem 0;
}

h1, h2, h3 {
color: var(--text); !important;
}

.hero {
padding: 6rem 2rem;
background: radial-gradient(circle at top, var(--gradient-start), var(--gradient-end));
border-radius: 12px;
color: white;
}

.hero h1 {
font-size: 3rem;
margin-bottom: 1rem;
}

.hero span {
color: var(--primary);
}

.hero p {
color: var(--muted);
font-size: 1.1rem;
max-width: 700px;
}

.btn {
display: inline-block;
padding: 0.75rem 1.5rem;
border-radius: 8px;
font-weight: 600;
margin-right: 1rem;
}

.btn-primary {
background: var(--primary);
color: white;
}

.btn-outline {
border: 1px solid var(--primary);
color: var(--primary);
}

.card {
background: var(--card);
padding: 1.5rem;
border-radius: 12px;
border: 1px solid var(--gradient-start);
height: 100%;
}

.cta {
background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
padding: 4rem 2rem;
border-radius: 12px;
text-align: center;
}
            

.st-emotion-cache-wfksaw {
//background: var(--gradient-start);
//border-radius: 10px;
}

.stButton > button {
background: var(--primary) !important;
color: white !important;
border-radius: 8px !important;
font-weight: 600 !important;
padding: 0.75rem 1.5rem !important;
border: none !important;
}
#MainMenu {visibility: hidden;}     /* Hides the '...' hamburger menu */
header {visibility: hidden;}        /* Hides the top bar */
footer {visibility: hidden;}        /* Hides 'Made with Streamlit' */
</style>
""", unsafe_allow_html=True)

# --- Navigation setup ---
PAGES = ["Home", "Create a request", "List requests", "Explore results", "Compute RT"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- Navbar ---
cols = st.columns(len(PAGES))
for i, page in enumerate(PAGES):
    if cols[i].button(page):
        st.session_state.page = page
        st.rerun()

st.markdown("---")

# --- Page: Home ---
if st.session_state.page == "Home":
    # ------------------ HERO ------------------
    st.markdown("""
    <div class="hero">
    <h1>Machine Unlearning <br><span>as a Service</span></h1>
    <p>
        Remove unwanted data, concepts, or identities from AI models —
        securely, efficiently, and without retraining from scratch.
    </p>
    </div>
    """, unsafe_allow_html=True)
    

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ------------------ FEATURES ------------------
    st.markdown("## Why Forgety?")
    col1, col2, col3, col4 = st.columns(4)

    features = [
        ("Fast & Efficient", "Selective unlearning without full retraining."),
        ("Regulation-Ready", "Designed for GDPR and right-to-be-forgotten."),
        ("Performance Preserving", "Retention examples maintain quality."),
        ("GPU-Powered", "Runs on our managed GPU infrastructure."),
    ]

    for col, (title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="card">
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ------------------ HOW IT WORKS ------------------
    st.markdown("## How It Works")

    st.markdown("""
    <ol>
    <li>Upload your model and data to forget</li>
    <li>Apply LoRA-based FADE and UCE unlearning</li>
    <li>Validate performance and compliance</li>
    <li>Deploy your updated model</li>
    </ol>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ------------------ USE CASES ------------------
    st.markdown("## Use Cases")
    uc1, uc2, uc3, uc4 = st.columns(4)

    use_cases = [
        ("GDPR Compliance", "Erase personal data safely."),
        ("Brand Safety", "Remove misaligned brand concepts."),
        ("Copyright Protection", "Unlearn protected styles."),
        ("Model Refinement", "Adapt to evolving requirements."),
    ]

    for col, (title, desc) in zip([uc1, uc2, uc3, uc4], use_cases):
        with col:
            st.markdown(f"""
            <div class="card">
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ------------------ CTA ------------------
    st.markdown("""
    <div id="contact" class="cta">
    <h2>Start Unlearning Today</h2>
    <p>A future-proof solution for companies relying on foundation models.</p>
    <br>
    </div>
    """, unsafe_allow_html=True)

# --- Page: Create a request ---
elif st.session_state.page == "Create a request":
    st.title("Create a Request")
    with st.form("request_form", clear_on_submit=True):
        experiment_name = st.text_input("Experiment Name (*)")
        model_base_name = st.text_input("Model Base Name (*)")
        concept_forget = st.text_input("Concept to Forget")
        concept_overwrite = st.text_input("Concept to Overwrite")
        concept_retain = st.text_input("Concept to Retain")
        unlearning_algorithm = st.selectbox(
            "Unlearning Algorithm (*)",
            ["Automatic (recommended)", "FADE", "Munba", "UCE"]
        )
        model_output_hf_id = st.text_input("Model Output HF ID (*)")
        dataset_file = st.file_uploader("Dataset (ZIP file)", type=["zip"])

        submitted = st.form_submit_button("Send")

        if submitted:
            # Validation
            missing_fields = []
            if not experiment_name:
                missing_fields.append("Experiment Name")
            if not model_base_name:
                missing_fields.append("Model Base Name")
            if not unlearning_algorithm:
                missing_fields.append("Unlearning Algorithm")
            if not model_output_hf_id:
                missing_fields.append("Model Output HF ID")

            if missing_fields:
                st.error("Missing required fields: " + ", ".join(missing_fields))
            else:
                try:
                    with st.spinner("Sending request..."):
                        response = requests.post(
                            "http://backend:80/v1/public-api-create-request",
                            data={
                                "customer_id": "demo-customer",
                                "experiment_name": experiment_name,
                                "model_base_name": model_base_name,
                                "concept_forget": concept_forget,
                                "concept_overwrite": concept_overwrite,
                                "concept_retain": concept_retain,
                                "unlearning_algorithm": unlearning_algorithm,
                                "model_output_hf_id": model_output_hf_id,
                            },
                            files={"dataset": (dataset_file.name, dataset_file, "application/zip")} if dataset_file else None,
                            timeout=180,
                        )
                    if response.status_code == 200:
                        st.success("Request sent successfully!")
                        st.session_state.page = "List requests"
                        st.rerun()
                    else:
                        st.error(f"Backend error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

# --- Page: List requests ---
elif st.session_state.page == "List requests":
    st.title("List of Requests")

    response = requests.get("http://backend:80/v1/public-api-read-requests", params={"customer_id": "demo-customer"})
    if response.status_code == 200:
        requests_data = response.json()
        if not requests_data:
            st.info("No requests found.")
        else:
            for req in requests_data:
                metrics: Optional[List[Dict[str, float]]] = req.get("metrics", None)
                text = f"**{req['experiment_name']}** - {req['model_base_name']}\n\n"
                text += f"Concept to Forget: {req.get('concept_forget', 'N/A')}\n\n"
                text += f"Concept to Overwrite: {req.get('concept_overwrite', 'N/A')}\n\n"
                text += f"Concept to Retain: {req.get('concept_retain', 'N/A')}\n\n"
                text += f"Unlearning Algorithm: {req['unlearning_algorithm']}\n\n"
                text += f"Model Output HF ID: {req['model_output_hf_id']}\n\n"
                text += f"Status: {req.get('status', 'N/A')}\n\n"
                if metrics:
                    text += f"Metrics: \n{metrics_to_markdown(metrics)}\n\n"
                with st.container():
                    st.markdown(text, help="Some extra info...")
                    st.markdown("---")

# --- Page: Explore results ---
elif st.session_state.page == "Explore results":
    st.title("Explore the results of a I-CARE compatible benchmark")
    model = st.selectbox(
        "Model",
        domain_model,
    )

    task = st.selectbox(
        "Task",
        domain_task,
    )

    # ------------------------------------------------------------------ #
    # Helper: metric glosses for the five key interference metrics shown   #
    # in Section B of the entity detail dialog.                            #
    # ------------------------------------------------------------------ #
    _KEY_METRICS: List[tuple] = [
        ("Emitter average clip diff", "avg CLIP distance: target vs all retained concepts"),
        ("Emitter worst interfered clip diff", "CLIP distance to the most interfered retained concept"),
        ("Receiver average clip diff", "avg CLIP loss seen by retained concepts disturbed by this entity"),
        ("Emitter minus receiver average clip diff", "target forgetting effect minus collateral interference"),
        ("Embedding specificity ratio", "directional specificity in DINOv2 space (>1 = targeted forgetting)"),
    ]

    @st.dialog("Entity details", width="large")  # type: ignore[misc]
    def show_details(row: Dict[str, Any], current_model: str, current_task: str) -> None:  # noqa: E501
        """Richer entity detail panel with five sections:
        A — Entity profile, B — Benchmark-computed metrics, Visual — RTs for all 3 methods,
        D — Latent space profile (EmbeddingUnlearningProfile, distil),
        C — Author-reported HF model card metrics, E — Provenance.
        """
        entity_name: str = row['name']
        st.markdown(f"## {entity_name}")

        # ----------------------------------------------------------------
        # Section A — Entity Profile
        # ----------------------------------------------------------------
        st.subheader("Entity Profile")
        profile_fields = {k: v for k, v in row.items() if not k.startswith('metric_') and k != 'name'}
        if profile_fields:
            profile_items = [{"Attribute": k, "Value": str(v)} for k, v in profile_fields.items() if v not in (None, "", "nan")]
            if profile_items:
                st.dataframe(profile_items, hide_index=True, use_container_width=True)
            else:
                st.info("No profile attributes available for this entity.")
        else:
            st.info("No profile attributes available for this entity.")

        # ----------------------------------------------------------------
        # Section B — Benchmark-computed metrics (all 3 methods)
        # ----------------------------------------------------------------
        st.subheader("Benchmark-computed metrics")
        st.caption("Produced by the I-CARE benchmark (vision-unlearning). Raw metric names preserved for reproducibility.")
        metric_cols: List[str] = [k for k in row.keys() if k.startswith('metric_')]
        methods_display = [("distil", "FADE/distil"), ("munba", "Munba"), ("uce", "UCE")]
        tab_labels = [label for _, label in methods_display]
        tabs = st.tabs(tab_labels)
        for tab, (method_key, _) in zip(tabs, methods_display):
            with tab:
                rows_b: List[Dict[str, str]] = []
                for me_name, gloss in _KEY_METRICS:
                    try:
                        col_name = choose_metric_column_interference_per_entity(method_key, me_name, metric_cols)  # type: ignore[arg-type]
                        val = row.get(col_name)
                        val_str = f"{val:.4f}" if isinstance(val, float) else str(val) if val is not None else "—"
                    except (ValueError, KeyError):
                        col_name = me_name.lower().replace(" ", "_")
                        val_str = "—"
                    rows_b.append({"Metric": me_name, "Value": val_str, "Description": gloss})
                st.dataframe(rows_b, hide_index=True, use_container_width=True)

        # ----------------------------------------------------------------
        # Visual — InterferenceVisualSummary for all 3 methods
        # (hardcoded clip_diff; each in an expander)
        # ----------------------------------------------------------------
        st.subheader("Interference Visual Summary")
        st.caption("All three unlearning methods, clip_diff interference metric.")
        for method_key, method_label in methods_display:
            with st.expander(f"{method_label}", expanded=False):
                show_rt_graceful(
                    template="InterferenceVisualSummary",
                    model=current_model,
                    task=current_task,
                    params={
                        "unlearning_algorithm": method_key,
                        "interference_pair": "clip_diff",
                        "entity": entity_name,
                    },
                )

        # ----------------------------------------------------------------
        # Section D — Latent Space Profile (EmbeddingUnlearningProfile, distil)
        # ----------------------------------------------------------------
        st.subheader("Latent Space Profile")
        st.caption("Embedding-space shift for this entity after forgetting (FADE/distil method).")
        with st.expander("Show latent space profile", expanded=False):
            show_rt_graceful(
                template="EmbeddingUnlearningProfile",
                model=current_model,
                task=current_task,
                params={
                    "unlearning_algorithm": "distil",
                    "entity": entity_name,
                },
            )

        # ----------------------------------------------------------------
        # Section C — Author-reported metrics (from HF model card)
        # ----------------------------------------------------------------
        st.subheader("Author-reported metrics (from HF model card)")
        st.caption(
            "Self-reported metrics from the HuggingFace model card of the unlearned model. "
            "Epistemologically distinct from the benchmark-computed metrics above — "
            "discrepancies between sections are scientifically interesting."
        )
        hf_dataset_url = "https://huggingface.co/datasets/LeonardoBenitez/VisionUnlearningEvaluationTestbeds"
        for method_key, method_label in methods_display:
            with st.expander(f"HF model card — {method_label}", expanded=False):
                try:
                    hf_resp = requests.get(
                        "http://backend:80/v1/public-api-entity-model-metrics",
                        params={"task": current_task, "entity": entity_name, "unlearning_algorithm": method_key},
                        timeout=30,
                    )
                    if hf_resp.status_code == 200:
                        hf_metrics: Dict[str, Any] = hf_resp.json()
                        if hf_metrics:
                            for k, v in hf_metrics.items():
                                st.markdown(f"- **{k}**: {v}")
                        else:
                            st.warning("Model card metrics not yet available for this entity/method.")
                    else:
                        st.warning("Model card metrics not yet available for this entity/method.")
                except Exception:
                    st.warning("Could not reach backend for HF model card metrics.")
                st.markdown(
                    f"[View full dataset on HuggingFace ↗]({hf_dataset_url})",
                    unsafe_allow_html=False,
                )

        # ----------------------------------------------------------------
        # Section E — Provenance
        # ----------------------------------------------------------------
        with st.expander("How this was computed"):
            st.markdown(
                "**Benchmark-computed metrics** (Section B) and the latent space profile (Section D) are produced by the "
                "[I-CARE benchmark](https://huggingface.co/datasets/LeonardoBenitez/VisionUnlearningEvaluationTestbeds) "
                "using the `vision_unlearning.benchmarks.I_care` package. "
                "The interference-per-entity computation runs `InterferencePerEntity(task=...).compute()`, "
                "which measures CLIP-based image quality changes across the forget and retain sets for every "
                "entity pair. DINOv2 embeddings are used for the latent space specificity ratio. "
                "All generated images and pre-computed results are stored in the HuggingFace dataset above."
            )

    try:
        with st.spinner("Loading results..."):
            response = requests.get(
                "http://backend:80/v1/public-api-read-interference-per-entity-all",
                timeout=60,
            )
        if response.status_code == 200:
            data = response.json()[task.lower()]

            event = st.dataframe(
                data,
                on_select="rerun",
                selection_mode="single-row",
            )

            selected_rows = event.selection.rows
            if selected_rows:
                row = data[selected_rows[0]]
                show_details(row, model, task)
        else:
            st.error(f"Backend error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
        
# --- Page: Compute RT ---
elif st.session_state.page == "Compute RT":
    st.title("Compute on-the-fly Result Template")
    template = st.selectbox(
        "Result Template (*)",
        rt_name_to_class.keys(),
    )

    model = st.selectbox(
        "Model",
        domain_model,
    )

    task = st.selectbox(
        "Task",
        domain_task,
    )

    with st.form("rt_form", clear_on_submit=True):
        params = {}

        for p in rt_name_to_params.get(template, []):
            if p in ["model", "task"]:
                continue  # already selected above

            if p == "unlearning_algorithm":
                params[p] = st.selectbox("Unlearning Algorithm", domain_unlearning_algorithm)

            elif p == "interference_entity_1":
                params[p] = st.selectbox("MetricInterferencePerEntity 1", domain_me)

            elif p == "interference_entity_2":
                params[p] = st.selectbox("MetricInterferencePerEntity 2", domain_me)

            elif p == "interference_entity":
                params[p] = st.selectbox("MetricInterferencePerEntity", domain_me)

            elif p == "interference_entity_list":
                val = st.multiselect("MetricInterferencePerEntity", domain_me)
                params[p] = ",".join(val) if val else None

            elif p == "attribute":
                params[p] = st.selectbox("Attribute", domain_attribute[task])

            elif p == "attribute_1":
                params[p] = st.selectbox("Attribute 1", domain_attribute[task])

            elif p == "attribute_2":
                params[p] = st.selectbox("Attribute 2", domain_attribute[task])

            elif p == "attribute_value":
                params[p] = st.text_input("Attribute Value (optional)")

            elif p == "attribute_list":
                val = st.multiselect("Attributes", domain_attribute[task])
                params[p] = ",".join(val) if val else None

            elif p == "interference_pair":
                params[p] = st.selectbox("MetricInterferencePerEntityPair", domain_mp)

            elif p == "similarity_metric":
                params[p] = st.selectbox("SimilarityBetweenEntities", domain_s)

            elif p == "latent_embedding":
                params[p] = st.selectbox("LatentEmbedding", domain_l)

            elif p == "entity":
                params[p] = st.selectbox("Entity", domain_entity[task])

            elif p == "entity_1":
                params[p] = st.selectbox("Entity 1", domain_entity[task])

            elif p == "entity_2":
                params[p] = st.selectbox("Entity 2", domain_entity[task])


        submitted = st.form_submit_button("Compute")
        if submitted:
            request_rt_to_backend(template, model, task, params)

# --- Error case ---
else:
    st.error("Unknown page selected.")


