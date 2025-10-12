import streamlit as st
import requests

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
        unlearning_algorithm = st.selectbox(
            "Unlearning Algorithm (*)",
            ["Automatic (recommended)", "SalUn", "FADE", "Munba"]
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
            if dataset_file is None:
                missing_fields.append("Dataset")

            if missing_fields:
                st.error("Missing required fields: " + ", ".join(missing_fields))
            else:
                files = {"dataset": dataset_file.getvalue()}
                data = {
                    "customer_id": "demo-customer",
                    "experiment_name": experiment_name,
                    "model_base_name": model_base_name,
                    "concept_forget": concept_forget,
                    "concept_overwrite": concept_overwrite,
                    "unlearning_algorithm": unlearning_algorithm,
                    "model_output_hf_id": model_output_hf_id,
                }
                
                with st.spinner("Sending request..."):
                    # TODO
                    st.success("Request sent successfully!")
                    st.session_state.page = "List requests"
                    st.rerun()

# --- Page: List requests ---
elif st.session_state.page == "List requests":
    st.title("List of Requests")

    requests_data = [
        {
            "experiment_name": "Modern art generation, remove John Snow works",
            "model_base_name": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "concept_forget": "John Snow",
            "concept_overwrite": "Vanilla modern art",
            "unlearning_algorithm": "FADE",
            "model_output_hf_id": "demo-customer/modern-art-no-john-snow",
            "status": "LAUNCHED",
            "metrics": "N/A"
        },
        {
            "experiment_name": "Prehistoric painting generator, forget cellphones",
            "model_base_name": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "concept_forget": "Someone holding a cellphone",
            "concept_overwrite": "Someone holding a rock",
            "unlearning_algorithm": "FADE",
            "model_output_hf_id": "demo-customer/prehistoric-art-no-cellphones",
            "status": "SUCCEEDED",
            "metrics": "[{'name': 'accuracy', 'value': 0.95}]"
        },
        {
            "experiment_name": "Prehistoric painting generator, forget cellphones",
            "model_base_name": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "concept_forget": None,
            "concept_overwrite": None,
            "unlearning_algorithm": "Munba",
            "model_output_hf_id": "demo-customer/prehistoric-art-no-cellphones",
            "status": "FAILED",
            "metrics": "[{}]"
        }
    ]
    for req in requests_data:
        with st.container():
            st.markdown(
                f"**{req['experiment_name']}** - {req['model_base_name']}\n\n"
                f"Concept to Forget: {req.get('concept_forget', 'N/A')}\n\n"
                f"Concept to Overwrite: {req.get('concept_overwrite', 'N/A')}\n\n"
                f"Unlearning Algorithm: {req['unlearning_algorithm']}\n\n"
                f"Model Output HF ID: {req['model_output_hf_id']}\n\n"
                f"Status: {req.get('status', 'N/A')}\n\n"
                f"Metrics: {req.get('metrics', 'N/A')}\n\n",
                help="Some extra info, whatever..."
            )
            st.markdown("---")
