from typing import List, Dict, Optional
import streamlit as st
import requests
import re

# TODO: all this metadata should be retrieved from the backend, where it is calculated from the files in huggingface
domain_unlearning_algorithm = ["FADE", "Munba", "UCE"]
domain_task = ["Breeds", "Scenes", "People"]
domain_attribute = {
    "Breeds": [
        'dataset_n_original',
        'group',
        'section',
        'country',
        'name_akc',
        'description',
        'temperament',
        'popularity',
        'min_height',
        'max_height',
        'min_weight',
        'max_weight',
        'min_expectancy',
        'max_expectancy',
        'group_akc',
        'grooming_frequency_value',
        'grooming_frequency_category',
        'shedding_value',
        'shedding_category',
        'energy_level_value',
        'energy_level_category',
        'trainability_value',
        'trainability_category',
        'demeanor_value',
        'demeanor_category',
        'URL',
        'Pronunciation',
        'Other Names',
        'Nickname',
        'Origin',
        'group_pawsome',
        'Size',
        'Male Height Min (in)',
        'Male Height Max (in)',
        'Male Height Min (cm)',
        'Male Height Max (cm)',
        'Female Height Min (in)',
        'Female Height Max (in)',
        'Female Height Min (cm)',
        'Female Height Max (cm)',
        'Male Weight Min (lbs)',
        'Male Weight Max (lbs)',
        'Male Weight Min (kg)',
        'Male Weight Max (kg)',
        'Female Weight Min (lbs)',
        'Female Weight Max (lbs)',
        'Female Weight Min (kg)',
        'Female Weight Max (kg)',
        'Coat Length',
        'Coat Type',
        'Double Coat',
        'Hypoallergenic',
        'Affection Rating',
        'Playfulness Rating',
        'Protectiveness Rating',
        'Territoriality Rating',
        'Prey Drive Rating',
        'Barking Rating',
        'Good with Children Rating',
        'Good with Adults Rating',
        'Good with Dogs Rating',
        'Good with Pets Rating',
        'Good with Strangers Rating',
        'Sociability Rating',
        'Sensitivity Rating',
        'Separation Anxiety Rating',
        'Energy Rating',
        'Intelligence Rating',
        'Mental Stimulation Rating',
        'Obedience Rating',
        'Trainability Rating',
        'Stubbornness Rating',
        'Attention Span',
        'Shedding Rating',
        'Grooming Rating',
        'Drooling Rating',
        'Lifespan Min',
        'Lifespan Max',
        'Health Rating',
        'Dental Issues Rating',
        'Ear Issues Rating',
        'Eye Issues Rating',
        'Owner Experience Rating',
        'First Time Owner',
        'Apartment Living Rating',
        'grooming_frequency_category_binary',
    ],
    "Scenes": [
        'sailing/ boating',
        'driving',
        'biking',
        'transporting things or people',
        'sunbathing',
        'vacationing/ touring',
        'hiking',
        'climbing',
        'camping',
        'reading',
        'studying/ learning',
        'teaching/ training',
        'research',
        'diving',
        'swimming',
        'bathing',
        'eating',
        'cleaning',
        'socializing',
        'congregating',
        'waiting in line/ queuing',
        'competing',
        'sports',
        'exercise',
        'playing',
        'gaming',
        'spectating/ being in an audience',
        'farming',
        'constructing/ building',
        'shopping',
        'medical activity',
        'working',
        'using tools',
        'digging',
        'conducting business',
        'praying',
        'fencing',
        'railing',
        'wire',
        'railroad',
        'trees',
        'grass',
        'vegetation',
        'shrubbery',
        'foliage',
        'leaves',
        'flowers',
        'asphalt',
        'pavement',
        'shingles',
        'carpet',
        'brick',
        'tiles',
        'concrete',
        'metal',
        'paper',
        'wood (not part of a tree)',
        'vinyl/ linoleum',
        'rubber/ plastic',
        'cloth',
        'sand',
        'rock/stone',
        'dirt/soil',
        'marble',
        'glass',
        'waves/ surf',
        'ocean',
        'running water',
        'still water',
        'ice',
        'snow',
        'clouds',
        'smoke',
        'fire',
        'natural light',
        'direct sun/sunny',
        'electric/indoor lighting',
        'aged/ worn',
        'glossy',
        'matte',
        'sterile',
        'moist/ damp',
        'dry',
        'dirty',
        'rusty',
        'warm',
        'cold',
        'natural',
        'man-made',
        'open area',
        'semi-enclosed area',
        'enclosed area',
        'far-away horizon',
        'no horizon',
        'rugged scene',
        'mostly vertical components',
        'mostly horizontal components',
        'symmetrical',
        'cluttered space',
        'scary',
        'soothing',
        'stressful',
        'dataset_n_original',
    ],
    "People": [
        'dataset_n_original',
        'birthyear',
        'gender',
        'occupation',
        'bplace_country',
        'hpi',
        'race',
        'occupation_simplified',
        'hpi_bin',
    ],
}
domain_entity = {
    "Breeds": [
        'dogo argentino',
        'griffon bruxellois dog',
        'griffon belge dog',
        'norwegian elkhound black dog',
        'norwegian elkhound grey dog',
        'tibetan terrier dog',
        'leonberger dog',
        'pekingese dog',
        'flat coated retriever dog',
        'bosnian and herzegovinian - croatian shepherd dog',
        'border terrier dog',
        'basenji dog',
        'american staffordshire terrier dog',
        'bouvier des ardennes dog',
        'bouvier des flandres dog',
        'nova scotia duck tolling retriever dog',
        'coton de tulear dog',
        'staffordshire bull terrier dog',
        'rottweiler dog',
        'chinese crested dog',
        'giant schnauzer dog',
        'chow chow dog',
        'great swiss mountain dog',
        'old english sheepdog',
        'lhasa apso dog',
        'miniature pinscher dog',
        'cairn terrier dog',
        'welsh corgi (cardigan) dog',
        'dogue de bordeaux',
        'shar pei dog',
        'bull terrier dog',
        'airedale terrier dog',
        'samoyed dog',
        'alaskan malamute dog',
        'scottish terrier dog',
        'australian cattle dog',
        'continental toy spaniel dog',
        'irish soft coated wheaten terrier dog',
        'english cocker spaniel dog',
        'bullmastiff dog',
        'portuguese water dog',
        'bulldog',
        'st. bernard dog',
        'akita dog',
        'bichon frise dog',
        'chesapeake bay retriever dog',
        'shiba dog',
        'belgian shepherd dog',
        'west highland white terrier dog',
        'newfoundland dog',
        'french bulldog',
        'maltese dog',
        'border collie dog',
        'miniature american shepherd dog',
        'chihuahua dog',
        'golden retriever dog',
        'pug dog',
        'english springer spaniel dog',
        'shetland sheepdog',
        'canarian warren hound dog',
        'bernese mountain dog',
        'boston terrier dog',
        'german shepherd dog',
        'norwegian lundehund dog',
        'miniature schnauzer dog',
        'berger de beauce dog',
        'cesky terrier dog',
        'finnish spitz dog',
        'finnish lapponian dog',
        "cirneco dell'etna dog",
        'pyrenean sheepdog - smooth faced',
        'sussex spaniel dog',
        'king charles spaniel dog',
        'cavalier king charles spaniel dog',
        'canaan dog',
        'skye terrier dog',
        'dandie dinmont terrier dog',
        'irish glen of imaal terrier dog',
        'komondor dog',
        'polish lowland sheepdog',
        'australian shepherd dog',
        'american water spaniel dog',
        'sealyham terrier dog',
        'kuvasz dog',
        'curly coated retriever dog',
        'puli dog',
        'irish water spaniel dog',
        'spanish water dog',
        'nederlandse kooikerhondje dog',
        'field spaniel dog',
        'affenpinscher dog',
        'lakeland terrier dog',
        'clumber spaniel dog',
        'bedlington terrier dog',
        'australian terrier dog',
        'tibetan mastiff dog',
        'norfolk terrier dog',
        'tibetan spaniel dog',
        'russian black terrier dog',
        'german spitz dog',
    ],
    "Scenes": [
        'abbey',
        'waterfall_cascade',
        'velodrome_outdoor',
        'volleyball_court_outdoor',
        'arena_hockey',
        'arena_basketball',
        'terrace_farm',
        'tree_farm',
        'tundra',
        'valley',
        'stone_circle',
        'volcano',
        'waterfall_cataract',
        'pavilion',
        'waterfall_fan',
        'waterfall_plunge',
        'watering_hole',
        'wave',
        'wheat_field',
        'waterfall_block',
        'bamboo_forest',
        'snowfield',
        'sea_cliff',
        'moor',
        'velodrome_indoor',
        'track_outdoor',
        'track_indoor',
        'tennis_court_outdoor',
        'athletic_field_outdoor',
        'badminton_court_indoor',
        'badminton_court_outdoor',
        'baseball_field',
        'basketball_court_indoor',
        'basketball_court_outdoor',
        'batters_box',
        'batting_cage_indoor',
        'batting_cage_outdoor',
        'boxing_ring',
        'bullpen',
        'football_field',
        'ice_skating_rink_indoor',
        'martial_arts_gym',
        'pitchers_mound',
        'soccer_field',
        'squash_court',
        'stadium_baseball',
        'stadium_football',
        'stadium_soccer',
        'tennis_court_indoor',
        'mountain',
        'mountain_path',
        'mountain_snowy',
        'observatory_indoor',
        'nursing_home',
        'packaging_plant',
        'pagoda',
        'palace',
        'pantry',
        'pier',
        'picnic_area',
        'piano_store',
        'physics_laboratory',
        'phone_booth',
        'pharmacy',
        'pet_shop',
        'jail_indoor',
        'pedestrian_overpass_outdoor',
        'patio',
        'particle_accelerator',
        'parlor',
        'parking_lot',
        'parking_garage_outdoor',
        'parking_garage_indoor',
        'parade_ground',
        'oast_house',
        'observatory_outdoor',
        'oasis',
        'office',
        'ocean',
        'orchard',
        'ski_slope',
        'outcropping',
        'pasture',
        'pond',
        'rainforest',
        'river',
        'rock_arch',
        'sandbar',
        'savanna',
        'park',
        'badlands',
        'ossuary',
        'organ_loft_exterior',
        'optician',
        'operating_room',
        'oilrig',
        'oil_refinery_outdoor',
        'office_cubicles',
        'office_building',
        'wrestling_ring_indoor',
    ],
    "People": [
        'George_W_Bush',
        'Colin_Powell',
        'Tony_Blair',
        'Donald_Rumsfeld',
        'Ariel_Sharon',
        'Junichiro_Koizumi',
        'John_Ashcroft',
        'Jacques_Chirac',
        'Serena_Williams',
        'Vladimir_Putin',
        'Gloria_Macapagal_Arroyo',
        'Arnold_Schwarzenegger',
        'Jennifer_Capriati',
        'Lleyton_Hewitt',
        'Laura_Bush',
        'Alejandro_Toledo',
        'Andre_Agassi',
        'Silvio_Berlusconi',
        'Tom_Ridge',
        'Megawati_Sukarnoputri',
        'Vicente_Fox',
        'Roh_Moo-hyun',
        'David_Beckham',
        'John_Negroponte',
        'Guillermo_Coria',
        'Mahmoud_Abbas',
        'Jack_Straw',
        'Juan_Carlos_Ferrero',
        'Ricardo_Lagos',
        'Gray_Davis',
        'Tom_Daschle',
        'Atal_Bihari_Vajpayee',
        'Winona_Ryder',
        'Tiger_Woods',
        'Lindsay_Davenport',
        'Naomi_Watts',
        'Pete_Sampras',
        'Jennifer_Lopez',
        'Jennifer_Aniston',
        'Carlos_Menem',
        'Angelina_Jolie',
        'Igor_Ivanov',
        'Julianne_Moore',
        'John_Howard',
        'Joschka_Fischer',
        'Nicole_Kidman',
        'Tim_Henman',
        'Lance_Armstrong',
        'Michael_Schumacher',
        'Jean_Charest',
        'Spencer_Abraham',
        'Venus_Williams',
        'Trent_Lott',
        'Halle_Berry',
        'Dominique_de_Villepin',
        'Meryl_Streep',
        'Pierce_Brosnan',
        'Andy_Roddick',
        'Norah_Jones',
        'Kim_Clijsters',
        'David_Nalbandian',
        'Roger_Federer',
        'James_Blake',
        'Britney_Spears',
        'Edmund_Stoiber',
        'Salma_Hayek',
        'Jackie_Chan',
        'Joe_Lieberman',
        'Jennifer_Garner',
        'Michael_Jackson',
        'Jeb_Bush',
        'Harrison_Ford',
        'Adrien_Brody',
        'Howard_Dean',
        'Rubens_Barrichello',
        'Anna_Kournikova',
        'Mike_Weir',
        'Mark_Philippoussis',
        'Ian_Thorpe',
        'Muhammad_Ali',
        'Kate_Hudson',
        'Colin_Farrell',
        'Ray_Romano',
        'Maria_Shriver',
        'Justin_Timberlake',
        'Bob_Hope',
        'Robert_Blake',
        'Amelia_Vega',
        'Clay_Aiken',
        'Zinedine_Zidane',
        'Valentino_Rossi',
        'Boris_Becker',
        'Elsa_Zylberstein',
        'Lance_Bass',
        'Natalie_Maines',
        'Ludivine_Sagnier',
        'George_Lopez',
        'Martina_McBride',
        'Michael_Chiklis',
        'Steffi_Graf'
    ],
}
domain_model = ["Stable Diffusion 1.4"]
domain_mp = ["DeltaClip", "DeltaBrisque", "RMSE", "SSIM"]
domain_me = [
    "Emitter worst interfered brisque diff",
    "Emitter worst interfered clip diff",
    "Emitter worst interfered rmse",
    "Emitter worst interfered ssim",
    "Emitter number of interfered worse than target brisque diff",
    "Emitter number of interfered worse than target clip diff",
    "Emitter number of interfered worse than target rmse",
    "Emitter number of interfered worse than target ssim",
    "Emitter number of interfered worse than zero clip diff",
    "Emitter average brisque diff",
    "Emitter average clip diff",
    "Emitter average rmse",
    "Emitter average ssim",
    "Receiver worst interfered brisque diff",
    "Receiver worst interfered clip diff",
    "Receiver worst interfered rmse",
    "Receiver worst interfered ssim",
    "Receiver number of interfered worse than target brisque diff",
    "Receiver number of interfered worse than target clip diff",
    "Receiver number of interfered worse than target rmse",
    "Receiver number of interfered worse than target ssim",
    "Receiver number of interfered worse than zero clip diff",
    "Receiver average brisque diff",
    "Receiver average clip diff",
    "Receiver average rmse",
    "Receiver average ssim",
    "Emitter minus receiver worst interfered brisque diff",
    "Emitter minus receiver worst interfered clip diff",
    "Emitter minus receiver worst interfered rmse",
    "Emitter minus receiver worst interfered ssim",
    "Emitter minus receiver number of interfered worse than target brisque diff",
    "Emitter minus receiver number of interfered worse than target clip diff",
    "Emitter minus receiver number of interfered worse than target rmse",
    "Emitter minus receiver number of interfered worse than target ssim",
    "Emitter minus receiver number of interfered worse than zero clip diff",
    "Emitter minus receiver average brisque diff",
    "Emitter minus receiver average clip diff",
    "Emitter minus receiver average rmse",
    "Emitter minus receiver average ssim",
]
domain_s = [
    "clip cosine similarity",
    "jacc similarity",
]
domain_l = [
    "ClipEmbedding",
]

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
    try:
        with st.spinner("Sending request..."):
            response = requests.get(
                "http://backend:80/v1/public-api-read-results",
                timeout=60,
            )
        if response.status_code == 200:
            st.success("Request sent successfully!")
            results_data = response.json()
            st.markdown(results_data.get("html", ""), unsafe_allow_html=True)
        else:
            st.error(f"Backend error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
        
# --- Page: Compute RT ---
elif st.session_state.page == "Compute RT":
    st.title("Compute on-the-fly Result Template")
    template = st.selectbox(
        "Result Template (*)",
        [
            "MetricMetricAlignment",
            "MetricSimilarityAlignment",
            "InterferenceMatrix",
            "SimilarityMatrix",
            "SignificantRelationshipNumerical",
            "SignificantRelationshipCategorical",
            "CountSignificantRelationship",
            "ImplicitAssociationTest",
            "MinimumCutInterference",
            "UnlearningVisualSummary",
            "InterferenceVisualSummary",
        ],
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
        unlearning_algorithm = None
        if template not in ["SimilarityMatrix"]:
            unlearning_algorithm = st.selectbox(
                "Unlearning Algorithm",
                domain_unlearning_algorithm,
            )

        me1 = None
        if template in ["MetricMetricAlignment"]:
            me1 = st.selectbox("MetricInterferencePerEntity 1", domain_me)

        me2 = None
        if template in ["MetricMetricAlignment"]:
            me2 = st.selectbox("MetricInterferencePerEntity 2", domain_me)

        me = None
        if template in ["SignificantRelationshipNumerical", "SignificantRelationshipCategorical"]:
            me = st.selectbox("MetricInterferencePerEntity", domain_me)

        me_list = None
        if template in ["CountSignificantRelationship"]:
            me_list = st.multiselect("MetricInterferencePerEntity", domain_me)
        
        a = None
        if template in ["SignificantRelationshipNumerical", "SignificantRelationshipCategorical"]:
            a = st.selectbox("Attribute", domain_attribute[task])

        a1 = None
        if template in ["ImplicitAssociationTest"]:
            a1 = st.selectbox("Attribute 1", domain_attribute[task])

        a2 = None
        if template in ["ImplicitAssociationTest"]:
            a2 = st.selectbox("Attribute 2", domain_attribute[task])

        a_value = None
        if template in ["SignificantRelationshipCategorical"]:
            a_value = st.text_input("Attribute Value (optional)")

        a_list = None
        if template in ["CountSignificantRelationship"]:
            a_list = st.multiselect("Attributes", domain_attribute[task])

        mp = None
        if template in ["MetricSimilarityAlignment", "InterferenceMatrix", "MinimumCutInterference", "InterferenceVisualSummary"]:
            mp = st.selectbox("MetricInterferencePerEntityPair", domain_mp)

        s = None
        if template in ["MetricSimilarityAlignment", "SimilarityMatrix"]:
            s = st.selectbox("SimilarityBetweenEntities", domain_s)

        l = None
        if template in ["ImplicitAssociationTest"]:
            l = st.selectbox("LatentEmbedding", domain_l)

        e1 = None
        if template in ["MinimumCutInterference"]:
            e1 = st.selectbox("Entity 1", domain_entity[task])

        e2 = None
        if template in ["MinimumCutInterference"]:
            e2 = st.selectbox("Entity 2", domain_entity[task])

        submitted = st.form_submit_button("Compute")
        if submitted:
            try:
                with st.spinner("Sending request..."):
                    response = requests.post(
                        "http://backend:80/v1/public-api-compute-rt",
                        data={
                            "template": template,
                            "params": {
                                "model": model,
                                "task": task,
                                "unlearning_algorithm": unlearning_algorithm,
                                "me1": me1,
                                "me2": me2,
                                "me": me,
                                "me_list": ",".join(me_list) if me_list else None,
                                "a": a,
                                "a1": a1,
                                "a2": a2,
                                "a_value": a_value,
                                "a_list": ",".join(a_list) if a_list else None,
                                "mp": mp,
                                "s": s,
                                "l": l,
                                "e1": e1,
                                "e2": e2,
                            }
                        },
                        timeout=180,
                    )
                if response.status_code == 200:
                    st.success("Request sent successfully!")
                    result_image = response.content
                    st.image(result_image, caption="Computed Result Template")
                else:
                    st.error(f"Backend error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
# --- Error case ---
else:
    st.error("Unknown page selected.")


