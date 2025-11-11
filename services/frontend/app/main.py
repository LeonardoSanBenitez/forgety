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
                            timeout=30,
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
                with st.container():
                    st.markdown(
                        f"**{req['experiment_name']}** - {req['model_base_name']}\n\n"
                        f"Concept to Forget: {req.get('concept_forget', 'N/A')}\n\n"
                        f"Concept to Overwrite: {req.get('concept_overwrite', 'N/A')}\n\n"
                        f"Concept to Retain: {req.get('concept_retain', 'N/A')}\n\n"
                        f"Unlearning Algorithm: {req['unlearning_algorithm']}\n\n"
                        f"Model Output HF ID: {req['model_output_hf_id']}\n\n"
                        f"Status: {req.get('status', 'N/A')}\n\n"
                        f"Metrics: {req.get('metrics', 'N/A')}\n\n",
                        help="Some extra info, whatever..."
                    )
                    st.markdown("---")
