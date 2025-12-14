from typing import List, Dict, Optional
import streamlit as st
import requests
import re


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


# Remove the "deploy" and "..." from the top
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}     /* Hides the '...' hamburger menu */
    header {visibility: hidden;}        /* Hides the top bar */
    footer {visibility: hidden;}        /* Hides 'Made with Streamlit' */
    </style>
""", unsafe_allow_html=True)

# --- Navigation setup ---
PAGES = ["Home", "Create a request", "List requests"]

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
    st.title("Home")
    st.write("Hi")

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
