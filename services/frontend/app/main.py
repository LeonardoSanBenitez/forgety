from typing import List, Dict, Optional
import streamlit as st
import requests
import re
import traceback



from vision_unlearning.benchmarks.I_care import domain_unlearning_algorithm, domain_task, domain_attribute, domain_entity, domain_model, domain_mp, domain_me, domain_s, domain_l, rt_name_to_class, rt_name_to_params



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

    try:
        with st.spinner("Sending request..."):
            response = requests.get(
                "http://backend:80/v1/public-api-read-interference-per-entity-all",
                timeout=60,
            )
        if response.status_code == 200:
            #st.success("Request sent successfully!")
            data = response.json()[task.lower()]
            
            event = st.dataframe(
                data,
                on_select="rerun",
                selection_mode="single-row",
            )

            selected_rows = event.selection.rows

            if selected_rows:
                row = data[selected_rows[0]]

                @st.dialog("Details")
                def show_details():
                    request_rt_to_backend(
                        template="InterferenceVisualSummary",
                        model=model,
                        task=task,
                        params={
                            "unlearning_algorithm": "distil",  # hardcoded...
                            "interference_pair": "clip_diff",  # hardcoded...
                            "entity": row['name'],
                        }
                    )

                show_details()
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


