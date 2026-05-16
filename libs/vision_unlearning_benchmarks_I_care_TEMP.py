# For installation simplicitry, this same file is in forgety, but afterwards will be moved ot vision unlearning
from __future__ import annotations
import os
import re
import base64
import pandas as pd
import numpy as np
import json
import io
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel
from typing import Literal, Tuple, Optional, Any, Dict, List

from typing import Literal, Tuple, List, Dict, Optional, Any
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, linregress, pearsonr, spearmanr
from typing import List, Dict, Any, Literal
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns

import logging
import sys
import shutil
import pickle
from huggingface_hub import hf_api, HfApi, snapshot_download
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] %(name)-8s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
)
logger = logging.getLogger('vision_unlearning.' + 'RT')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(FORMATTER)
logger.addHandler(stdout_handler)
logger.setLevel(logging.INFO)


################################################################################################
#### new code to be moved to vision_unlearning
from huggingface_hub import hf_api, HfApi, snapshot_download, hf_hub_url, hf_hub_download
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from typing import Optional
import requests


def huggingface_dataset_exists(
    dataset_repository: str,
    dataset_config: str,
    token: Optional[str],
) -> bool:
    """
    Checks whether a folder exists in a Hugging Face dataset repository.

    Example:
        dataset_repository="username/my_dataset"
        dataset_config="configs/en"

    Works without listing the whole repository.
    """

    url = (
        f"https://huggingface.co/api/datasets/"
        f"{dataset_repository}/tree/main/{dataset_config.replace(os.sep, '/')}"
    )

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers)

    if response.status_code == 404:
        return False

    response.raise_for_status()

    # Existing folders return a JSON array of entries.
    return isinstance(response.json(), list)


def huggingface_dataset_file_exists(
    dataset_repository: str,
    dataset_path: str,
    token: Optional[str],
) -> bool:
    """
    Checks if a specific file exists in a Hugging Face dataset repository.

    :param dataset_repository: e.g. "username/dataset_name"
    :param dataset_path: full path in repo (e.g. "config/file.jsonl")
    :param token: HF token (can be None for public repos)
    :return: True if file exists, False otherwise
    Efficiently checks if a file exists in a Hugging Face dataset repo without listing the entire repository.
    Could be done more efficiently if we use a new version of the lib, see https://chatgpt.com/share/69edd525-d008-832d-8a0c-ec4560a4fe3b

    """
    url = hf_hub_url(
        repo_id=dataset_repository,
        filename=dataset_path,
        repo_type="dataset",
    )
    #print('url:', url, flush=True)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.head(url, headers=headers)
    return response.status_code in (200, 302, 303, 307)


def huggingface_dataset_file_download(
    folder_datasets: str,
    dataset_repository: str,
    file_path: str,
    token: str,
    folder_cache: str = '/tmp/huggingface_cache',
) -> None:
    '''
    Download a single file from a dataset in Hugging Face Hub.
    
    Args:
        folder_datasets: Local directory where datasets are stored.
        dataset_repository: Hugging Face dataset repository ID
        file_path: Full path of the file within the repository (e.g., "config/data.jsonl")
        token: Hugging Face authentication token
        folder_cache: Cache directory for downloads
    
    The file will be saved at os.path.join(folder_datasets, file_path)
    '''
    os.makedirs(folder_datasets, exist_ok=True)
    os.makedirs(folder_cache, exist_ok=True)
    
    # Download to cache
    cached_path = hf_hub_download(
        repo_id=dataset_repository,
        filename=file_path,
        repo_type="dataset",
        token=token,
        cache_dir=folder_cache,
    )

    # Copy from cache to final folder
    target_path = os.path.join(folder_datasets, file_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy2(cached_path, target_path)




def _encode_image_file(img_path: str, max_dim: int = 1024) -> str:
    '''
    Downsample / reduce resolution to limit size before encoding
    '''
    assert os.path.exists(img_path), f"Image file not found at {img_path}"
    with Image.open(img_path) as im:
        # Convert to RGB to ensure compatibility with JPEG
        if im.mode != 'RGB':
            im = im.convert('RGB')
        if max(im.size) > max_dim:

            scale = max_dim / max(im.size)
            new_size = (int(im.size[0] * scale), int(im.size[1] * scale))
            im = im.resize(new_size, Image.LANCZOS)
            #print(f"Resized image from {im.size} to {new_size} to limit size before encoding.")
        buf = io.BytesIO()
        im.save(buf, format='PNG', quality=85, optimize=True)
        image_bytes = buf.getvalue()
    return base64.b64encode(image_bytes).decode('ascii')

def _decode_image(image_data: str) -> io.BytesIO:
    assert isinstance(image_data, str), f"Expected image data to be a base64 string, but got {type(image_data)}"
    return io.BytesIO(base64.b64decode(image_data))

####

import json
import numpy as np
import pandas as pd
import shap

def explanation_to_dict(expl):
    return {
        "values": expl.values.tolist() if expl.values is not None else None,
        "base_values": (
            expl.base_values.tolist()
            if isinstance(expl.base_values, np.ndarray)
            else expl.base_values
        ),
        "data": expl.data.tolist() if expl.data is not None else None,
        "feature_names": list(expl.feature_names) if expl.feature_names is not None else None,
        "output_names": list(expl.output_names) if expl.output_names is not None else None,
    }

def dict_to_explanation(d):
    return shap.Explanation(
        values=np.array(d["values"]) if d["values"] is not None else None,
        base_values=np.array(d["base_values"]) if d["base_values"] is not None else None,
        data=np.array(d["data"]) if d["data"] is not None else None,
        feature_names=d["feature_names"],
        output_names=d["output_names"],
    )

# Every artifact should be abstracted by a OO class, instead of just a loosely connected set of functions
# This class should also handle automatically fetching the underlying data from huggingface
# For now both styles can coexist, but we should refactor all rest of code to use just OO
# This follows basically the same logic as a RT... Maybe both should inherit from a class "Artifact" or something like that
class InterferencePerEntity(BaseModel):
    task: type_task = 'people'
    base_folder: str = 'assets'
    remote_repository_name: str = 'LeonardoBenitez/VisionUnlearningEvaluationTestbeds'
    save_outputs: bool = True
    recompute_if_exists: bool = False
    # This class deprecates: save_interference_per_entity, get_interference_per_entity_path

    def _get_data_path_remote(self) -> str:
        return f'interference_per_entity_{self.task}.json'


    def _get_data_path_local(self) -> str:
        return os.path.join(self.base_folder, self._get_data_path_remote())


    def compute(self) -> List[Dict[str, Any]]:
        if not self.recompute_if_exists and os.path.exists(self._get_data_path_local()):  # Local
            with open(self._get_data_path_local(), "r", encoding="utf-8") as f:
                data: List[Dict[str, Any]] = json.load(f)
        elif not self.recompute_if_exists and huggingface_dataset_file_exists(  # Remote
            self.remote_repository_name,
            self._get_data_path_remote(),
            token=os.getenv('HF_TOKEN'),
        ):
            #print('going the remote option', flush=True)
            huggingface_dataset_file_download(
                folder_datasets=self.base_folder,
                dataset_repository=self.remote_repository_name,
                file_path=self._get_data_path_remote(),
                token=os.getenv('HF_TOKEN'),
            )
            assert os.path.exists(self._get_data_path_local())
            #print('downloaded', flush=True)
            with open(self._get_data_path_local(), "r", encoding="utf-8") as f:
                data: List[Dict[str, Any]] = json.load(f)
        else:  # Compute from scratch
            data = self._compute_from_scratch()
            if self.save_outputs:
                os.makedirs(os.path.dirname(self._get_data_path_local()), exist_ok=True)
                with open(self._get_data_path_local(), "w", encoding="utf-8") as f:
                    json.dump(data, f)
        assert type(data) == list, f"Expected a dict in the json file, but got {type(data)}"
        assert len(data) > 0  # == 100
        assert all(isinstance(item, dict) for item in data)
        return data

def _compute_from_scratch(self) -> List[Dict[str, Any]]:
    raise NotImplementedError()  # TODO

########################################################################################################
### Old code moved here for convience (do not modify!)

def huggingface_dataset_file_upload(
    file_path: str,
    dataset_repository: str,
    dataset_path: str,
    token: str,
):
    '''
    Upload a single file to a specific dataset config in Hugging Face Hub.
    @param dataset_path: full name of the file in the repository, including the config folder (e.g., "my_config/my_file.jsonl")
    '''
    assert os.path.exists(file_path)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=dataset_path,
        repo_id=dataset_repository,
        repo_type='dataset',
        token=token,
    )

def huggingface_dataset_download(
    folder_datasets: str,
    dataset_repository: str,
    dataset_config: str,
    token: str,
    clean: bool = False,
    folder_cache: str = '/tmp/huggingface_cache',
    clean_cache: bool = False,
):
    '''
    @param clean: If True, the folder will be deleted before downloading
    '''
    folder_dataset = os.path.join(folder_datasets, dataset_config)
    if clean:
        if os.path.exists(folder_dataset):
            shutil.rmtree(folder_dataset)
    if os.path.exists(folder_dataset):
        logger.info('Dataset already exists locally, skipping download')
        return
    os.makedirs(folder_dataset)

    folder_cache_dataset = os.path.join(folder_cache, dataset_repository, dataset_config)
    os.makedirs(folder_cache_dataset, exist_ok=True)

    # Download to cache
    repo_path = snapshot_download(
        repo_id=dataset_repository,
        repo_type="dataset",
        token=token,
        allow_patterns=f"{dataset_config}/*",
        cache_dir=folder_cache,
    )

    # Copy from cache to final folder
    for root, _, files in os.walk(os.path.join(repo_path, dataset_config)):
        for file in files:
            source_path = os.path.join(root, file)
            if os.path.islink(source_path):
                source_path = os.path.join(root, os.readlink(source_path))
            target_path = os.path.join(folder_dataset, os.path.relpath(os.path.join(root, file), start=os.path.join(repo_path, dataset_config)))
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)

    # Remove cache
    if clean_cache:
        shutil.rmtree(repo_path)


def get_interference_per_entity_path(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    base_folder: str = 'assets',
) -> str:
    #return f"assets/interference_per_entity_{task}.json"
    return os.path.join(base_folder, f'interference_per_entity_{task}.json')

def get_interference_per_entity(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    max_identities: int = 100,
) -> List[Dict[str, Any]]:
    assert os.path.exists(get_interference_per_entity_path(task))
    with open(get_interference_per_entity_path(task), "r", encoding="utf-8") as f:
        metadata_filtered = json.load(f)
    assert isinstance(metadata_filtered, list)
    assert len(metadata_filtered) == max_identities
    return metadata_filtered


def save_interference_per_entity(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    metadata_filtered: List[Dict[str, Any]],
) -> None:
    with open(get_interference_per_entity_path(task), "w", encoding="utf-8") as f:
        json.dump(metadata_filtered, f, indent=4)


def get_interference_per_pair_path(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    base_folder: str = 'assets',
) -> str:
    return os.path.join(base_folder, 'datasets', f'interferences_caused_by_{task}_{index}_{method}_{num_train_epochs}.json')


def get_interference_per_pair(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    max_identities: int = 100,
    base_folder: str = 'assets',
) -> Dict[str, Dict[str, float]]:
    # TODO: maybe this function should first check locally if the file exists, and if not, check in huggingface if the file exists there, and just then return an error if neighter?
    assert os.path.exists(get_interference_per_pair_path(task, index, method, num_train_epochs, base_folder)), "Caused interferences by this entity were not computed yet"
    with open(get_interference_per_pair_path(task, index, method, num_train_epochs, base_folder), 'r') as f:
        interference_per_pair = json.load(f)
    assert isinstance(interference_per_pair, dict)
    assert len(interference_per_pair) == max_identities
    return interference_per_pair


def get_metadata_filtered_path(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    base_folder: str = 'assets',
) -> str:
    return os.path.join(base_folder, f"metadata_{task}_2_enriched_filtered.json")


def get_metadata_filtered(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    base_folder: str = 'assets'
) -> List[Dict[str, Any]]:
    with open(get_metadata_filtered_path(task, base_folder=base_folder), "r", encoding="utf-8") as f:
        metadata_filtered = json.load(f)
    assert isinstance(metadata_filtered, list)
    return metadata_filtered


def get_generated_dataset_folder(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    target: str,
    base_folder: str = 'assets',
) -> str:
    # By convention, I'm passing here the preprocessed target... TODO change?
    return os.path.join(base_folder, "datasets", f"generated_{task}_{target}_{method}_{num_train_epochs:03d}")


def get_target_overwrite(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    method: Literal['munba', 'uce', 'distil'],
    target: str,
) -> Tuple[str, str]:
    '''
    @return preprocessed target, target_overwrite
    '''
    # TODO THIS SHOULD USE  get_target_preprocessed FOR THE TARGET!!!!
    if task == 'people':
        # target does NOT need to have an article,for example: picture of brad pitt

        # target_race = metadata_filtered[index]['race'].replace('indian_middleEastern_latinoHispanic', 'middle eastern') # enum: white, asian, black, indian_middleEastern_latinoHispanic
        # target_gender = 'male' if metadata_filtered[index]['gender']=='M' else 'female'  # Enum[M, F]
        # article = 'an' if (target_race[0].lower() in 'aeiou') else 'a'
        # target_overwrite = f"{article} {target_race} {target_gender}"  # For munba this is only the retain concept for final evaluation, there is no overwriting
        target_overwrite = 'a child'
    elif task == 'breeds':
        # target does needs to have an article,for example: picture of a poodle
        target_overwrite = 'a cat'
        article = 'an' if (target[0].lower() in 'aeiou') else 'a'
        #target = re.sub(r'\bdog\b', '', target, flags=re.IGNORECASE)
        target = f"{article} {target}"
    elif task == 'scenes':
        # target does needs to have an article,for example: picture of a phone_booth
        target_overwrite = 'the moon'
        article = 'an' if (target[0].lower() in 'aeiou') else 'a'
        target = f"{article} {target} scene"

    else:
        raise NotImplementedError()

    target = target.replace('_', ' ')
    target = re.sub(r'\s+', ' ', target).strip()

    assert isinstance(target_overwrite, str)
    assert isinstance(target, str)
    assert len(target) >= 3

    return target, target_overwrite

def get_generated_dataset_file(
    lora_state: Literal['on', 'off'],
    seed: int,
    prompt: str,
) -> str:
    return f'{lora_state}_{seed:02}_{prompt}.png'


#################################################
# Metadata about the benchmark
# TODO: all this metadata should be calculated from the files in huggingface
# Defining which RTs we have and which values are valid should be some process of "discovery"

# Petty formatted values
domain_unlearning_algorithm = ["FADE", "Munba", "UCE"]
domain_task = ["Breeds", "Scenes", "People"]
domain_attribute = {
    "Breeds": [
        #'dataset_n_original',
        'group',
        'section',
        'country',
        #'name_akc',
        #'description',
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
        #'URL',
        #'Pronunciation',
        #'Other Names',
        #'Nickname',
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
        #'dataset_n_original',
    ],
    "People": [
        #'dataset_n_original',
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
domain_mp = ["Delta Clip", "Delta Brisque", "RMSE", "SSIM"]
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
    "Clip Cosine Similarity",
    "Jacc Similarity",
]
domain_l = [
    "Clip Embedding",
]

# Types (as they appear in the code/files)
type_unlearning_algorithm = Literal["distil", "munba", "uce"]
type_task = Literal["breeds", "scenes", "people"]
type_model = Literal["sd1.4"]
type_mp = Literal["brisque_diff", "clip_diff", "rmse", "ssim"]
type_me = Literal[
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
type_s = Literal[
    "clip",
    "jacc",
]

type_l = Literal[
    "clip_embedding",
]

type_regression_algorithm = Literal[
    "linear_regression",
    "random_forest",
]

# And converting between them
GUI_TO_BACKEND = {
    "unlearning_algorithm": {
        "FADE": "distil",
        "Munba": "munba",
        "UCE": "uce",
    },
    "task": {
        "Breeds": "breeds",
        "Scenes": "scenes",
        "People": "people",
    },
    "model": {
        "Stable Diffusion 1.4": "sd1.4",
    },
    "interference_pair": {
        "Delta Clip": "clip_diff",
        "Delta Brisque": "brisque_diff",
        "RMSE": "rmse",
        "SSIM": "ssim",
    },
    "similarity_metric": {
        "Clip Cosine Similarity": "clip",
        "Jacc Similarity": "jacc",
    },
    "latent_embedding": {
        "Clip Embedding": "clip_embedding",
    },
}


type_direction = Literal["↑", "↓"]

mp_to_direction: Dict[type_mp, type_direction] = {  # Higher =  more interference
    "brisque_diff": "↓",
    "clip_diff": "↑",  # zero = no change. Negative = generation actually got better
    "rmse": "↓",
    "ssim": "↑",
}
s_to_direction: Dict[type_s, type_direction] = {
    "clip": "↑",
    "jacc": "↑",
}
# me_to_direction can... be infered?


task_to_attributes_of_interest = {
    "breeds": [
        "grooming_frequency_category_binary",  # (a continuous attribute describing how often brushing is required; discretized into two bins based on quartiles)
        "supergroup",  # (categorical, with values such as retrievers and terriers)
    ],
    "scenes": [
        "sports",
        "natural",
    ],
    "people": [
        "hpi_bin",
        "occupation_simplified",
    ]
}


# The method's hyperparameter is specially problematic to map/find, because i initially thought it should
# be configurable and then I chagned my mind. Sometimes it can be infered (like in `choose_metric_column_interference_per_entity`),
# Also, in the actual slurm scripts they are hardcoded all over
# But sometimes I use this hardcoded mapping:
unlearning_algorithm_to_epochs = {
    'breeds': {
        'distil': 100,
        'munba': 100,
        'uce': 0,
    },
    'scenes': {
        'distil': 100,
        'munba': 100,
        'uce': 0,
    },
    'people': {
        'distil': 400,
        'munba': 200,
        'uce': 0,
    },
}

def choose_metric_column_interference_per_entity(
    unlearning_algorithm: type_unlearning_algorithm,
    interference_entity: type_me,
    metric_cols: List[str],
) -> str:
    """
    The columns of the interference per entity file are not named in a way that is easy to generate given `unlearning_algorithm` and `interference_entity`, so we need to search for the right one.
    We assume there is only one match, and we assert it. If there are no matches or more than one match, we raise an error.

    The names look like this:
        'metric_distil_400_emitter_minus_receiver_worst_interfered_ssim (↓)',
       'metric_distil_400_emitter_minus_receiver_number_of_interfered_worse_than_target_brisque_diff (↓)',
       'metric_distil_400_emitter_minus_receiver_number_of_interfered_worse_than_target_clip_diff (↓)',
       'metric_distil_400_emitter_minus_receiver_number_of_interfered_worse_than_target_rmse (↓)',
       'metric_distil_400_emitter_minus_receiver_number_of_interfered_worse_than_target_ssim (↓)',
       'metric_distil_400_emitter_minus_receiver_number_of_interfered_worse_than_zero_clip_diff (↓)',
       'metric_distil_400_emitter_minus_receiver_average_brisque_diff (↓)',
       'metric_distil_400_emitter_minus_receiver_average_clip_diff (↑)',
       'metric_uce_000_emitter_minus_receiver_average_rmse (↓)',
       'metric_munba_100_emitter_minus_receiver_average_ssim (↑)',
    
    TODO: these names are defined in `4. Compute interference per entity.ipynb`. There should be a central way of defining them.
    """
    pattern = f"metric_{unlearning_algorithm}_[^_]*_{interference_entity.lower().replace(' ', '_')} .*"
    matching_cols = [col for col in metric_cols if re.match(pattern, col)]
    if len(matching_cols) == 0:
        raise ValueError(f'No metric column found for unlearning_algorithm={unlearning_algorithm} and interference_entity={interference_entity}')
    elif len(matching_cols) > 1:
        raise ValueError(f'Multiple metric columns found for unlearning_algorithm={unlearning_algorithm} and interference_entity={interference_entity}: {matching_cols}')
    return matching_cols[0]


def convert_params_from_gui_to_backend(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert GUI values to backend literal values.
    Unknown keys are passed through unchanged.
    None stays None.
    """
    converted_params: Dict[str, Any] = {}

    for key, value in params.items():
        if value is None:
            converted_params[key] = None
            continue

        mapping = GUI_TO_BACKEND.get(key)
        if mapping is not None:
            converted_params[key] = mapping.get(value, value)
        else:
            converted_params[key] = value

    return converted_params


class InvalidAttributeTypeError(ValueError):
    pass


class InsufficientSamplesError(ValueError):
    pass


##################################################
# Actual RTs
class ResultTemplate(BaseModel):
    recompute_if_exists: bool = False
    save_outputs: bool = True
    base_folder: str = 'assets'
    remote_repository_name: str = 'LeonardoBenitez/VisionUnlearningEvaluationTestbeds'

    
    def _serialize_parameters(self) -> str:
        raise NotImplementedError()

    def _get_data_path_remote(self) -> str:
        return os.path.join("results", self.__class__.__name__.replace('ResultTemplate', ''), f"{self._serialize_parameters()}.json")

    def _get_data_path_local(self) -> str:
        return os.path.join(self.base_folder, self._get_data_path_remote())

    @classmethod
    def _fig_to_bytes(cls, fig: Figure) -> bytes:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer.getvalue()

    def _compute_from_scratch(self) -> dict | list:
        raise NotImplementedError()

    def compute(self) -> dict:
        if not self.recompute_if_exists and os.path.exists(self._get_data_path_local()):  # Local
            with open(self._get_data_path_local(), "r", encoding="utf-8") as f:
                data: dict  = json.load(f)
        elif not self.recompute_if_exists and huggingface_dataset_file_exists(  # Remote
            self.remote_repository_name,
            self._get_data_path_remote(),
            token=os.getenv('HF_TOKEN'),
        ):
            #print('going the remote option', flush=True)
            huggingface_dataset_file_download(
                folder_datasets=self.base_folder,
                dataset_repository=self.remote_repository_name,
                file_path=self._get_data_path_remote(),
                token=os.getenv('HF_TOKEN'),
            )
            assert os.path.exists(self._get_data_path_local())
            #print('downloaded', flush=True)
            with open(self._get_data_path_local(), "r", encoding="utf-8") as f:
                data: dict  = json.load(f)
        else:  # Compute from scratch
            data = self._compute_from_scratch()
            if self.save_outputs:
                os.makedirs(os.path.dirname(self._get_data_path_local()), exist_ok=True)
                with open(self._get_data_path_local(), "w", encoding="utf-8") as f:
                    json.dump(data, f)

        assert type(data) == dict, f"Expected a dict in the json file, but got {type(data)}"
        assert 'result' in data, f"Expected 'result' key in the json file, but got {list(data.keys())}"
        assert type(data['result']) in [dict, list], f"Expected 'result' to be a dict or list, but got {type(data['result'])}"
        return data


class ResultTemplateMetricMetricAlignment(ResultTemplate):
    """
    Measures how strongly two *MetricInterferencePerEntity* metrics are correlated.

    **Arguments:** `m`, `t`, `u`, `m_e1`, `m_e2`.
    **Result:** Pearson p-value, Spearman p-value, Pearson correlation, scatter plot.
    **Interpretation:** quantitative; the higher the correlation, the lower the need to
    calculate both metrics for this specific choice of `m`, `t`, and `u`.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_entity_1: type_me
    interference_entity_2: type_me



class ResultTemplateMetricSimilarityAlignment(ResultTemplate):
    """
    To what degree similar *entities* interfere more with each other.

    Formalized in `ap:prediction`, which also proposes its natural expansion to a
    multivariable and non-linear predictive regression.

    **Arguments:** `m`, `t`, `u`, `m_p`, `s`.
    **Result:** Pearson p-value, Spearman p-value, Pearson correlation, scatter plot.
    **Interpretation:** quantitative; if this value is high, interference between two
    *entities* can be approximated by *similarity* (which is cheaper to compute for any
    new *entity*). Equivalently, the amount of "transmission wires" can be summarized
    by this single *similarity* function.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    similarity_metric: type_s
    significance_threshold: float = 0.05

    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}_{self.similarity_metric}"

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (6, 5), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        fig, ax = plt.subplots(figsize=figsize)

        sns.scatterplot(
            x=data['result']['x'],
            y=data['result']['y'],
            ax=ax
        )

        sns.regplot(
            x=data['result']['x'],
            y=data['result']['y'],
            scatter=False,
            ax=ax
        )

        ax.set_xlabel(f"Interference $m_p$: {data['metadata']['interference_pair'].replace('_', ' ').title()} ({data['metadata']['interference_pair_direction']})", fontsize=8)
        ax.set_ylabel(f"Similarity $s$: {data['metadata']['similarity_metric'].replace('_', ' ').title()} ({data['metadata']['similarity_metric_direction']})", fontsize=8)

        ax.set_title(
            f"Task: {data['metadata']['task'].title()}\n"
            f"Method: {data['metadata']['unlearning_algorithm'].title()}\n"
            f"Pearson correlation: {data['result']['pearson_statistic']:.3f} (p-value: {data['result']['pearson_pvalue']:.3f})\n"
            f"Spearman correlation: {data['result']['spearman_statistic']:.3f} (p-value: {data['result']['spearman_pvalue']:.3f})",
            fontsize=10
        )

        plt.tight_layout(pad=0.5)

        if return_fig:
            return fig, ax
        else:
            plt.show()


    def _compute_from_scratch(self, exclude_diagonal: bool = True) -> dict:
        # TODO: redo the same logic, without converting to df in the middle
        df1 = pd.DataFrame(ResultTemplateInterferenceMatrix(
            model = self.model,
            task = self.task,
            unlearning_algorithm = self.unlearning_algorithm,
            interference_pair = self.interference_pair
        ).compute()['result'])
        df1.set_index('emitter', inplace=True)

        df2 = pd.DataFrame(ResultTemplateSimilarityMatrix(
            model = self.model,
            task = self.task,
            unlearning_algorithm = self.unlearning_algorithm,
            similarity_metric = self.similarity_metric
        ).compute()['result'])
        df2.set_index('emitter', inplace=True)

        if df1.shape != df2.shape:
            raise ValueError("DataFrames must have the same shape.")
        if not np.all(df1.index == df1.columns):
            raise ValueError("DataFrames must be square with matching indices and columns.")
        if not np.all(df1.index == df2.index):
            raise ValueError("DataFrames must have the same index and columns.")
        if not np.all(df1.columns == df2.columns):
            raise ValueError("DataFrames must have the same index and columns.")
        labels = df1.index.to_list()

        # Prepare data
        # Each cell ij becomes a row {'c1': df1_ij, 'c2': df2_ij}
        # index are the labelsi_to_labelj
        df_prepared = pd.DataFrame(columns=['c1', 'c2'])
        for label_i in labels:
            for label_j in labels:
                if exclude_diagonal and (label_i == label_j):
                    continue
                value1 = df1.loc[label_i, label_j]
                value2 = df2.loc[label_i, label_j]
                df_prepared = pd.concat([df_prepared, pd.DataFrame({'c1': [value1], 'c2': [value2]}, index=[f'{label_i}_to_{label_j}'])])
        assert df_prepared.shape[0] == (df1.shape[0] * df1.shape[1] - (df1.shape[0] if exclude_diagonal else 0))
        assert pd.api.types.is_numeric_dtype(df_prepared['c1']), f"{self.interference_pair} must be numeric"
        assert pd.api.types.is_numeric_dtype(df_prepared['c2']), f"{self.similarity_metric} must be numeric"

        # Significance tests
        df_prepared.dropna(inplace=True)
        x = df_prepared['c1'].astype(float).to_list()
        y = df_prepared['c2'].astype(float).to_list()
        pearson_res = pearsonr(x, y)
        spearman_res = spearmanr(x, y)

        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_pair': self.interference_pair,
                'similarity_metric': self.similarity_metric,
                'interference_pair_direction': mp_to_direction[self.interference_pair],
                'similarity_metric_direction': s_to_direction[self.similarity_metric],
                'significance_threshold': self.significance_threshold,
            },
            'result': {
                'x': x,
                'y': y,
                'pearson_statistic': pearson_res.statistic,
                'pearson_pvalue': pearson_res.pvalue,
                'spearman_statistic': spearman_res.statistic,
                'spearman_pvalue': spearman_res.pvalue,
                'significant': bool(pearson_res.pvalue < self.significance_threshold or spearman_res.pvalue < self.significance_threshold),
            }
        }
        return data


class ResultTemplateMetricSimilarityAlignmentMulti(ResultTemplate):
    """
    Multi-input Single-output Regression Generalization of ResultTemplateMetricSimilarityAlignment (see also Appendix E, adapted from the multi-output setting).
    Also, the interpretability and feature engineering aspects are improved.

    ---

    We consider a fixed *model* \(m\), *task* \(t\), and *unlearning method* \(u\), which are omitted for brevity.

    The objective is to quantify whether interference between *entities* is aligned with their *similarity*, i.e., to what degree similar *entities* interfere more with each other.

    For every ordered pair of distinct *entities* \(e_i, e_j \in t\) with \(i \neq j\), we observe several *SimilarityBetweenEntities* measures, indexed by superscripts \(\ell = 1, 2, \dots, |S|\), and a single *MetricInterferencePerEntityPair* target \(m_p(e_i,e_j)\).

    Each ordered pair \((e_i, e_j)\) is therefore treated as one data point with feature vector

    $$
    \mathbf{X}_{ij}
    =
    \big(
    s^{(1)}(e_i, e_j),
    \dots,
    s^{(|S|)}(e_i, e_j)
    \big)
    $$

    and scalar target

    $$
    Y_{ij}
    =
    m_p(e_i, e_j).
    $$

    The resulting dataset is

    $$
    \mathcal{D}
    =
    \{
    (\mathbf{X}_{ij}, Y_{ij})
    \mid
    e_i, e_j \in t,\ i \neq j
    \}.
    $$

    From this dataset, a regression model can be estimated using standard regression procedures with appropriate validation.

    In the linear case,

    $$
    Y_{ij}
    =
    \beta_0
    +
    \sum_{\ell=1}^{|S|}
    \beta_{\ell}
    X^{(\ell)}_{ij}
    +
    \varepsilon_{ij}.
    $$

    Given a specific *entity* \(e_i\) whose removal is considered, similarities

    $$
    X^{(\ell)}_{ij}
    =
    s^{(\ell)}(e_i, e_j)
    $$

    can be computed for all remaining *entities* \(e_j \in t\). The fitted model then yields predictions

    $$
    \hat{Y}_{ij}
    =
    f(\mathbf{X}_{ij}),
    $$

    which approximate the expected interference on each receiver *entity*.


    Furthermore, the concept of *similarity* may also encode several forms of practical data engineering. For example, one may define:
    - a distinct *similarity* function for each *attribute*, or
    - a *similarity* function based only on the attributes of the emitter entity.

    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    similarity_metric_list: List[type_s]
    significance_threshold: float = 0.05
    include_attribute_diff_similarity: bool = True
    include_attribute_value_similarity: bool = True
    regression_algorithm: type_regression_algorithm = 'linear_regression'
    random_state: int = 42
    test_size: float = 0.3

    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}_{'_'.join(self.similarity_metric_list)}_{int(self.include_attribute_diff_similarity)}_{int(self.include_attribute_value_similarity)}_{self.regression_algorithm}"

    def _get_partial_path_local(self):
        return self._get_data_path_local() + '.partial'

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (6, 15), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        explanations = dict_to_explanation(data['result']['shap_explanations'])

        fig, ax = plt.subplots(figsize=figsize)
        y_true = np.asarray(data['result']['y_test_true'], dtype=float)
        y_pred = np.asarray(data['result']['y_test_pred'], dtype=float)

        ax.scatter(y_true, y_pred, alpha=0.7)
        min_val = float(np.nanmin([y_true.min(), y_pred.min()]))
        max_val = float(np.nanmax([y_true.max(), y_pred.max()]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('True value')
        ax.set_ylabel('Predicted value')
        ax.set_title(f"True vs Predicted (MAPE: {data['result'].get('mape_test', float('nan')):.4f})")
        ax.grid(True, alpha=0.3)


        shap.plots.bar(explanations)
        shap.plots.beeswarm(explanations)

        if return_fig:
            return fig, ax
        plt.show()
        return None

    def _compute_from_scratch(self, exclude_diagonal: bool = True, entity_col: str = 'name') -> dict:
        # Gather precomputed data
        metadata_filtered = get_metadata_filtered(self.task)
        df_mp = pd.DataFrame(ResultTemplateInterferenceMatrix(
            model = self.model,
            task = self.task,
            unlearning_algorithm = self.unlearning_algorithm,
            interference_pair = self.interference_pair
        ).compute()['result'])
        df_mp.set_index('emitter', inplace=True)

        df_s_list = []
        for similarity_metric in self.similarity_metric_list:
            df_s = pd.DataFrame(ResultTemplateSimilarityMatrix(
                model = self.model,
                task = self.task,
                unlearning_algorithm = self.unlearning_algorithm,
                similarity_metric = similarity_metric
            ).compute()['result'])
            df_s.set_index('emitter', inplace=True)
            df_s_list.append(df_s)

        for df_s in df_s_list:
            if df_mp.shape != df_s.shape:
                raise ValueError("DataFrames must have the same shape.")
            if not np.all(df_mp.index == df_mp.columns):
                raise ValueError("DataFrames must be square with matching indices and columns.")
            if not np.all(df_mp.index == df_s.index):
                raise ValueError("DataFrames must have the same index and columns.")
            if not np.all(df_mp.columns == df_s.columns):
                raise ValueError("DataFrames must have the same index and columns.")

        # Prepare data
        # Each cell ij becomes a row
        # One col for the target (the metric-interference-per-pair, entry ij of df_mp), then one col per feature (each similarity + engineered features)
        # index are the labelsi_to_labelj
        labels = df_mp.index.to_list()
        columns = [self.interference_pair] + self.similarity_metric_list
        for attribute in task_to_attributes_of_interest[self.task]:
            if self.include_attribute_diff_similarity:
                columns.append(f'is_{attribute}_same')
            if self.include_attribute_value_similarity:
                columns.append(f'emitter_{attribute}_value')
                columns.append(f'receiver_{attribute}_value')
        df_prepared = pd.DataFrame(columns=columns)
        for label_emitter in labels:
            for label_receiver in labels:
                if exclude_diagonal and (label_emitter == label_receiver):
                    continue
                
                row_dict = {self.interference_pair: df_mp.loc[label_emitter, label_receiver]}
                for idx, similarity_metric in enumerate(self.similarity_metric_list):
                    row_dict[similarity_metric] = df_s_list[idx].loc[label_emitter, label_receiver]
                
                # Feature engineering
                # This logic is very similar to jacc_metric_score
                row_emitter = next((row for row in metadata_filtered if row[entity_col] == label_emitter), None)
                row_receiver = next((row for row in metadata_filtered if row[entity_col] == label_receiver), None)
                if row_emitter is None or row_receiver is None:
                    raise ValueError(f"Entities {label_emitter} and/or {label_receiver} not found in metadata")
                if set(row_emitter.keys()) != set(row_receiver.keys()):
                    raise ValueError(f"Entities {label_emitter} and {label_receiver} must have the same attributes")
                
                for attribute in task_to_attributes_of_interest[self.task]:
                    assert attribute in row_emitter, f"Attribute {attribute} not found in metadata for entity {label_emitter}"
                    assert attribute in row_receiver, f"Attribute {attribute} not found in metadata for entity {label_receiver}"
                    assert type(row_emitter[attribute]) == type(row_receiver[attribute]), f"Attribute {attribute} must have the same type for both entities {label_emitter} and {label_receiver}"
                    if type(row_emitter[attribute]) in [np.float64, float]:
                        logger.warning(f"Equality comparison for float attribute {attribute} may be unreliable")
                    if self.include_attribute_diff_similarity:
                        row_dict[f'is_{attribute}_same'] = float(row_emitter[attribute] == row_receiver[attribute])
                    if self.include_attribute_value_similarity:
                        row_dict[f'emitter_{attribute}_value'] = row_emitter[attribute]
                        row_dict[f'receiver_{attribute}_value'] = row_receiver[attribute]
                
                row_df = pd.DataFrame([row_dict], index=[f'{label_emitter}_to_{label_receiver}'])
                assert list(row_df.columns) == list(df_prepared.columns), f"Expected columns {df_prepared.columns}, but got {row_df.columns}"
                assert row_df.shape == (1, len(columns))
                df_prepared = pd.concat([df_prepared, row_df])
        
        assert df_prepared.shape[0] == (df_mp.shape[0] * df_mp.shape[1] - (df_mp.shape[0] if exclude_diagonal else 0))
        for col in self.similarity_metric_list:
            assert col in df_prepared.columns, f"Expected column {col} in df_prepared, but got {df_prepared.columns}"
            assert pd.api.types.is_numeric_dtype(df_prepared[col]), f"Expected column {col} to be numeric, but got {df_prepared[col].dtype}"
        df_prepared.dropna(inplace=True)


        # Split 70-30 train-test
        target_col = self.interference_pair
        X = df_prepared.drop(columns=[target_col])
        y = pd.to_numeric(df_prepared[target_col], errors='coerce')
        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ],
            remainder='drop'
        )
        
        # Fit regression model
        if self.regression_algorithm == 'random_forest':
            regressor = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
        else:
            regressor = LinearRegression()

        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', regressor),
        ])
        model_pipeline.fit(X_train, y_train)

        trained_model_path = self._get_partial_path_local()
        os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)
        with open(trained_model_path, 'wb') as f:
            pickle.dump(model_pipeline, f)
        
        # Analyze errors
        y_pred_train = model_pipeline.predict(X_train)
        y_pred_test = model_pipeline.predict(X_test)
        r2_train = float(r2_score(y_train, y_pred_train))
        r2_test = float(r2_score(y_test, y_pred_test))        

        feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()

        # TODO global F-test:
        # whether the model explains variance better than a null/intercept-only model

        # Shap
        X_sample = X.sample(n=min(1000, len(X)), random_state=self.random_state)
        X_sample_preprocessed = model_pipeline.named_steps['preprocessor'].transform(X_sample)
        X_sample_preprocessed_df = pd.DataFrame(X_sample_preprocessed, columns=[feature.split('__')[1] for feature in feature_names])
        if self.regression_algorithm == 'random_forest':
            explainer = shap.TreeExplainer(model_pipeline.named_steps['regressor'])
        elif self.regression_algorithm == 'linear_regression':
            explainer = shap.LinearExplainer(model_pipeline.named_steps['regressor'], X_sample_preprocessed, feature_perturbation='interventional')
        else:
            raise ValueError(f"Unsupported regression algorithm: {self.regression_algorithm}")

        explanations = explainer(X_sample_preprocessed_df)





        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_pair': self.interference_pair,
                'similarity_metric_list': self.similarity_metric_list,
                'interference_pair_direction': mp_to_direction[self.interference_pair],
                'similarity_metric_directions': [s_to_direction[sim] for sim in self.similarity_metric_list],
                'significance_threshold': self.significance_threshold,
                'include_attribute_diff_similarity': self.include_attribute_diff_similarity,
                'include_attribute_value_similarity': self.include_attribute_value_similarity,
                'regression_algorithm': self.regression_algorithm,
                'trained_model_path': trained_model_path,
            },
            'result': {
                'n_train': int(len(X_train)),
                'n_test': int(len(X_test)),
                'r2_train': r2_train,
                'r2_test': r2_test,
                'rmse_train': float(root_mean_squared_error(y_train, y_pred_train)),
                'rmse_test': float(root_mean_squared_error(y_test, y_pred_test)),
                'features': feature_names,
                'y_test_true': y_test.tolist(),
                'y_test_pred': y_pred_test.tolist(),
                'shap_explanations': explanation_to_dict(explanations),
            }
        }
        return data


class ResultTemplateSignificantRelationshipNumerical(ResultTemplate):
    """
    Measures whether two numerical attributes are significantly correlated.

    Formalized in `ap:rt_relationship`.

    **Arguments:** `m`, `t`, `u`, `m_e`, `a`.
    **Result:** Pearson p-value, Spearman p-value, Pearson correlation, scatter plot.
    **Interpretation:** qualitative; the researcher should decide if it is ethical or
    desirable that this *attribute* propagates interferences.

    **Pearson test**
        Use when you want to measure a **linear** relationship.
        **Assumptions:**
          * Both variables are **continuous**
          * Relationship is **linear**
          * **Bivariate normality** (both jointly Gaussian)
          * **Homoscedasticity** (constant variance)
          * **No strong outliers** (very sensitive)
        **Detects:** linear correlation only
        **Fails when:** relationship is monotonic but non-linear, or heavy outliers exist
    
    **Spearman test**
        Use when you want to measure a **monotonic** relationship (not necessarily linear) or data is non-Gaussian.
        **Assumptions:**
          * Variables are at least **ordinal**
          * Relationship is **monotonic** (increasing or decreasing)
          * **No distributional assumptions**
          * **Robust to outliers**
        **Detects:** any monotonic trend (linear or curved)
        **Fails when:** relationship is non-monotonic (e.g., U-shaped)
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_entity: type_me
    attribute: str
    significance_threshold: float = 0.05


    def _get_data_path_remote(self) -> str:
        return os.path.join("results", self.__class__.__name__.replace('ResultTemplate', ''), f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_entity}_{self.attribute}.json")


    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (6, 5), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        fig, ax = plt.subplots(figsize=figsize)

        method_name_pretty = data['metadata']['unlearning_algorithm'].title()
        metric_name_pretty = f"{data['metadata']['interference_entity']} ({data['metadata']['interference_entity_direction']})"
        attribute_name_pretty = data['metadata']['attribute'].replace('_', ' ').title()

        sns.scatterplot(
            x=data['result']['x'],
            y=data['result']['y'],
            ax=ax
        )

        sns.regplot(
            x=data['result']['x'],
            y=data['result']['y'],
            scatter=False,
            ax=ax
        )

        ax.set_xlabel(attribute_name_pretty, fontsize=8)
        ax.set_ylabel(metric_name_pretty, fontsize=8)

        ax.set_title(
            f"Task: {data['metadata']['task'].title()}\n"
            f"Metric: {metric_name_pretty}\n"
            f"Method: {method_name_pretty}\n"
            f"Attribute: {attribute_name_pretty}\n"
            f"Pearson p-value: {data['result']['pearson_pvalue']:.03}\n"
            f"Spearman p-value: {data['result']['spearman_pvalue']:.03}",
            fontsize=10
        )

        plt.tight_layout(pad=0.5)

        if return_fig:
            return fig, ax
        else:
            plt.show()


    def _compute_from_scratch(self) -> dict:
        # This part is common with the categorical version
        interference_per_entity_path: str = get_interference_per_entity_path(self.task)
        if not os.path.exists(interference_per_entity_path):
            raise FileNotFoundError(f"Interference per entity file not found at {interference_per_entity_path}. Please compute it before runnign this RT.")
        df = pd.read_json(interference_per_entity_path)
        metric_cols: List[str] = list(filter(lambda c: c.startswith('metric_'), df.columns))
        assert all(df[metric].dtype == np.float64 or df[metric].dtype == np.int64 for metric in metric_cols)
        for col in metric_cols:
            df[col] = df[col].astype(float)

        df_temp = df.dropna(subset=[self.attribute])
        df_temp_shape_after_attributes = df_temp.shape[0]
        if df_temp.shape[0] != df.shape[0]:
            logger.warning(f'Attribute {self.attribute} has NaN values, dropped {df.shape[0] - df_temp.shape[0]} rows')

        chosen_metric_col: str = choose_metric_column_interference_per_entity(self.unlearning_algorithm, self.interference_entity, metric_cols)
        df_temp = df.dropna(subset=[chosen_metric_col])
        if df_temp.shape[0] != df_temp_shape_after_attributes:
            logger.debug(f'Metric {chosen_metric_col} has NaN values, dropped {df_temp_shape_after_attributes - df_temp.shape[0]} rows')

        # this part is specific to numeric attributes
        attribute_type = type(df_temp[self.attribute].iloc[0])
        if attribute_type not in [int, np.int64, float, np.float64]:
            raise InvalidAttributeTypeError(f'Attribute {self.attribute} is not numerical, has type {attribute_type}')
        df_temp.loc[:, self.attribute] = df_temp.loc[:, self.attribute].astype(float)

        x = df_temp[self.attribute].astype(float).to_list()
        y = df_temp[chosen_metric_col].astype(float).to_list()
        pearson_res = pearsonr(x, y)
        spearman_res = spearmanr(x, y)

        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_entity': self.interference_entity,
                'attribute': self.attribute,
                'interference_entity_direction': chosen_metric_col.split(' ')[1][1],
                'chosen_metric_col': chosen_metric_col,
                'significance_threshold': self.significance_threshold,
            },
            'result': {
                'x': x,
                'y': y,
                'pearson_statistic': pearson_res.statistic,
                'pearson_pvalue': pearson_res.pvalue,
                'spearman_statistic': spearman_res.statistic,
                'spearman_pvalue': spearman_res.pvalue,
                'significant': bool(pearson_res.pvalue < self.significance_threshold or spearman_res.pvalue < self.significance_threshold),
            }
        }
        return data


class ResultTemplateSignificantRelationshipCategorical(ResultTemplate):
    """
    Statistical significance of the average `MetricInterferencePerEntity` across all
    *entities*, when grouped by each of its values.

    Formalized in `ap:rt_relationship`.

    **Arguments:** `m`, `t`, `u`, `m_e`, `a`, optional `filterAttributeValue`.
    **Result:** ANOVA p-value, Kruskal-Wallis p-value, average value of `m_e` grouped
    by each value of `a`, grouped boxplot.
    **Interpretation:** qualitative; similar to
    *SignificantRelationshipNumerical*. The optional argument
    *filterAttributeValue* restricts which emitter *entities* are included, allowing
    the analysis of interference flow distribution, such as whether politicians cause
    more interference to other politicians than artists cause to other artists.

    **ANOVA**
        Use when you want to test if **group means differ** across **3+ independent groups** under parametric assumptions.
        **Assumptions:**
          * Dependent variable is **continuous**
          * Groups are **independent**
          * **Normality** within each group
          * **Homoscedasticity** (equal variances)
          * No strong **outliers**
        **Hypothesis:**
          * H₀: all group means are equal
          * H₁: at least one mean differs
        **Detects:** differences in **means**
        **Fails when:** heavy skew, unequal variances, small n with non-Gaussian data

    **Kruskal-Wallis**
        Use when you want to test if **group distributions differ** without parametric assumptions.
        **Assumptions:**
          * Dependent variable is **ordinal or continuous**
          * Groups are **independent**
          * **Same shaped distributions** (only medians should differ for clean interpretation)
          * No normality or equal-variance requirement
        **Hypothesis:**
          * H₀: all group distributions are equal
          * H₁: at least one group differs
        **Detects:** differences in **medians / distributions**
        **Fails when:** distributions differ in shape (then result is ambiguous)
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_entity: type_me
    attribute: str
    attribute_value: Optional[str|int] = None
    min_samples_per_category: int = 5
    significance_threshold: float = 0.05


    def _get_data_path_remote(self) -> str:
        return os.path.join("results", self.__class__.__name__.replace('ResultTemplate', ''), f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_entity}_{self.attribute}_{self.attribute_value}.json")


    @classmethod
    def plot(cls, data: dict, extra_title: str = '', figsize: Tuple[int, int] = (6, 5), return_fig: bool =False) -> Optional[Tuple[Figure, plt.Axes]]:
        fig, ax = plt.subplots(figsize=figsize)

        method_name_pretty = data['metadata']['unlearning_algorithm'].title()
        metric_name_pretty = f"{data['metadata']['interference_entity']} ({data['metadata']['interference_entity_direction']})"
        attribute_name_pretty = data['metadata']['attribute'].replace('_', ' ').title()

        sns.boxplot(
            x=data['result']['x'],
            y=data['result']['y'],
            ax=ax,
            showfliers=False,
        )

        sns.stripplot(
            x=data['result']['x'],
            y=data['result']['y'],
            ax=ax,
            color='black',
            alpha=0.5,
        )
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel(attribute_name_pretty, fontsize=8)
        ax.set_ylabel(metric_name_pretty, fontsize=8)

        ax.set_title(
            f"Metric: {metric_name_pretty}\n"
            f"Attribute: {attribute_name_pretty}\n"
            f"Method: {method_name_pretty}\n"
            f"{extra_title}"
            f"ANOVA p-value: {data['result']['anova_pvalue']:.03}\n"
            f"Kruskal-Wallis p-value: {data['result']['kruskal_pvalue']:.03}",
            fontsize=10
        )

        plt.tight_layout(pad=0.5)

        if return_fig:
            return fig, ax
        else:
            plt.show()

    def _compute_from_scratch(self) -> dict:
        interference_per_entity_path: str = get_interference_per_entity_path(self.task)
        if not os.path.exists(interference_per_entity_path):
            raise FileNotFoundError(f"Interference per entity file not found at {interference_per_entity_path}. Please compute it before runnign this RT.")
        df = pd.read_json(interference_per_entity_path)
        metric_cols: List[str] = list(filter(lambda c: c.startswith('metric_'), df.columns))
        assert all(df[metric].dtype == np.float64 or df[metric].dtype == np.int64 for metric in metric_cols)
        for col in metric_cols:
            df[col] = df[col].astype(float)

        df_temp = df.dropna(subset=[self.attribute])
        df_temp_shape_after_attributes = df_temp.shape[0]
        if df_temp.shape[0] != df.shape[0]:
            logger.warning(f'Attribute {self.attribute} has NaN values, dropped {df.shape[0] - df_temp.shape[0]} rows')

        chosen_metric_col: str = choose_metric_column_interference_per_entity(self.unlearning_algorithm, self.interference_entity, metric_cols)
        df_temp = df.dropna(subset=[chosen_metric_col])
        if df_temp.shape[0] != df_temp_shape_after_attributes:
            logger.debug(f'Metric {chosen_metric_col} has NaN values, dropped {df_temp_shape_after_attributes - df_temp.shape[0]} rows')

        # this part is specific to categorical attributes
        attribute_type = df_temp[self.attribute].dtype
        if attribute_type != object:
            raise InvalidAttributeTypeError(f'Attribute {self.attribute} is not categorical, has type {attribute_type}')

        categories: List[str] = df_temp[self.attribute].unique().tolist()
        metric_per_category: List[List[float]] = [df_temp[df_temp[self.attribute] == c][chosen_metric_col].to_list() for c in categories]
        if any(len(vals) < self.min_samples_per_category for vals in metric_per_category):
            raise InsufficientSamplesError(f"Attribute {self.attribute} has insufficient samples in at least one category")

        anova_res = f_oneway(*metric_per_category)
        kruskal_res = kruskal(*metric_per_category)

        x = df_temp[self.attribute].astype(str).to_list()
        y = df_temp[chosen_metric_col].astype(float).to_list()

        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_entity': self.interference_entity,
                'attribute': self.attribute,
                'attribute_value': self.attribute_value,
                'interference_entity_direction': chosen_metric_col.split(' ')[1][1],
                'chosen_metric_col': chosen_metric_col,
                'significance_threshold': self.significance_threshold,
            },
            'result': {
                'x': x,
                'y': y,
                'anova_statistic': anova_res.statistic,
                'anova_pvalue': anova_res.pvalue,
                'kruskal_statistic': kruskal_res.statistic,
                'kruskal_pvalue': kruskal_res.pvalue,
                'significant': bool(anova_res.pvalue < self.significance_threshold or kruskal_res.pvalue < self.significance_threshold),
            }
        }
        return data


class ResultTemplateCountSignificantRelationship(ResultTemplate):
    """
    Number of significant relationships across all combinations of *attributes* and
    *MetricInterferencePerEntity*.

    **Arguments:** `m`, `t`, `u`, list of `m_e`, list of `a`.
    **Result:** integer, list of significances.
    **Interpretation:** quantitative; the lower the better. Since the attributes for
    which it is ethical to propagate interference are constant across all *models* and
    *methods*, a higher value directly implies a higher number of ethical violations,
    that is, a larger number of "transmission wires" in a given task effectively used
    by this *method* and *model*.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm_list: List[type_unlearning_algorithm]
    interference_entity_list: List[type_me]
    attribute_list: List[str]
    top_n: int = 10


    def _serialize_parameters(self) -> str:
        unlearning_algorithms_str = ','.join([ua.__name__ for ua in self.unlearning_algorithm_list])
        interference_entities_str = ','.join(self.interference_entity_list)
        attributes_str = ','.join(self.attribute_list)
        return f"{self.model}_{self.task}_{unlearning_algorithms_str}_{interference_entities_str}_{attributes_str}"

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (6, 5), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        pass

    def _compute_from_scratch(self) -> dict:
        results = []
        for unlearning_algorithm in self.unlearning_algorithm_list:
            for interference_entity in self.interference_entity_list:
                for attribute in self.attribute_list:
                    try:
                        data = ResultTemplateSignificantRelationshipCategorical(model=model, task=task, unlearning_algorithm=unlearning_algorithm, interference_entity=interference_entity, attribute=attribute).compute()
                    except InvalidAttributeTypeError:
                        data = ResultTemplateSignificantRelationshipNumerical(model=model, task=task, unlearning_algorithm=unlearning_algorithm, interference_entity=interference_entity, attribute=attribute).compute()
                    except InsufficientSamplesError:
                        continue
                    except Exception as e:
                        logger.warning(f'Combination {model}, {task}, {unlearning_algorithm}, {interference_entity}, {attribute} failled with {e}')
                        assert 1==0#continue
                    results.append([model, task, unlearning_algorithm, interference_entity, attribute, data['result']['significant']])
        df = pd.DataFrame(results, columns=['model', 'task', 'unlearning_algorithm', 'interference_entity', 'attribute', 'significant'])
        #TODO
        print(df[df['task']=='people'].groupby('attribute').sum()['significant'].sort_values(ascending=False))
        print(df[df['task']=='people'].groupby('unlearning_algorithm').sum()['significant'].sort_values(ascending=False))
        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm_list': self.unlearning_algorithm_list,
                'interference_entity_list': self.interference_entity_list,
                'attribute_list': self.attribute_list,
            },
            'result': {
                'grouped_by_unlearning_algorithm': {},
                'grouped_by_attribute': {},
                'grouped_by_interference_entity': {},
            }
        }
        return data
'''TEMP
results = []
for model in list(type_model.__args__): 
    for task in ['people']:#list(type_task.__args__):
        for unlearning_algorithm in list(type_unlearning_algorithm.__args__):
            for interference_entity in list(type_me.__args__):
                for attribute in domain_attribute[task.capitalize()]:
                    try:
                        data = ResultTemplateSignificantRelationshipCategorical(model=model, task=task, unlearning_algorithm=unlearning_algorithm, interference_entity=interference_entity, attribute=attribute).compute()
                    except InvalidAttributeTypeError:
                        data = ResultTemplateSignificantRelationshipNumerical(model=model, task=task, unlearning_algorithm=unlearning_algorithm, interference_entity=interference_entity, attribute=attribute).compute()
                    except InsufficientSamplesError:
                        continue
                    except Exception as e:
                        logger.warning(f'Combination {model}, {task}, {unlearning_algorithm}, {interference_entity}, {attribute} failled with {e}')
                        assert 1==0#continue
                    results.append([model, task, unlearning_algorithm, interference_entity, attribute, data['result']['significant']])
                    print('.', end='')
                print('')
            print('---')
df = pd.DataFrame(results, columns=['model', 'task', 'unlearning_algorithm', 'interference_entity', 'attribute', 'significant'])
df.head()
print(df.shape)
print(df[df['task']=='people'].groupby('attribute').sum()['significant'].sort_values(ascending=False))
print(df[df['task']=='people'].groupby('unlearning_algorithm').sum()['significant'].sort_values(ascending=False))
'''

class ResultTemplateImplicitAssociationTest(ResultTemplate):
    """
    Measures how the strength of automatic associations `B` between two pairs of
    *entities* changes after unlearning.

    **Arguments:** `m`, `t`, `u`, `a_1`, `a_2`, `l`.
    **Result:** `|a| x |a|` real-valued tensor `ΔB`.
    **Interpretation:** qualitative; a human should decide whether it is ethical or
    desirable for the unlearning process to cause this change in implicit association
    between the chosen *attributes*.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    attribute_1: str
    attribute_2: str
    latent_embedding: type_l


class ResultTemplateMinimumCutInterference(ResultTemplate):
    """
    Interprets a *task* as a directed weighted graph and computes the minimum cut separating two *entities*
    As a consequence of the max-flow min-cut theorem, it directly follows that the minimum cut is the smallest influence whose removal eliminates every directed influence path from $e_1$ to $e_2$.
    Based on this, we conjecture that if we need to unlearn $e_1$ while minimizing harm to $e_2$, then the ideal intervention in the unlearning process is to increase the preservation of the emitter-side nodes. More intuitively, we can think of this intervention as "blocking the interference path," as performed in electrical circuits to protect sensitive components (such as ground partitioning, shielding traces, among others.
    **Arguments:** $m$, $t$, $u$, $e_1$, $e_2$, $m_p$.
    **Result:** list of *entities* (corresponding to the emitter-side nodes).
    **Interpretation:** qualitative; small set of nodes through which most of the interference from $e_1$ propagates to $e_2$.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    entity_1: str
    entity_2: str


class ResultTemplateUnlearningVisualSummary(ResultTemplate):
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm


class ResultTemplateInterferenceVisualSummary(ResultTemplate):
    """
    Compared generated images for 9 identities: target, 4 worst (excluding target), 4 best
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    entity: Optional[str] = None  # Either entity or entity_index should be provided, but not both. Entity has priority over entity_index.
    entity_index: Optional[int] = None
    seed: int = 42
    images_max_dim: int = 124

    def _resolve_entity(self):
        '''
        Ensures both entity andentity_index are filled.
        Modifies in place
        At the end, both are set and consistent with each other
        '''
        metadata_filtered = get_metadata_filtered(self.task)
        if not self.entity:
            if self.entity_index is None:
                raise ValueError("Either entity or entity_index must be provided.")
            self.entity = metadata_filtered[self.entity_index]['name']
        else:
            expected_entity_index = next((i for i, item in enumerate(metadata_filtered) if item['name'] == self.entity), None)
            if expected_entity_index is None:
                raise ValueError(f"Entity '{self.entity}' not found in metadata.")

            if self.entity_index is None:
                self.entity_index = expected_entity_index
            else:
                if self.entity_index != expected_entity_index:
                    raise ValueError(f"Provided entity_index {self.entity_index} does not match the index of the provided entity '{self.entity}' in metadata, which is {expected_entity_index}.")
        assert type(self.entity) == str, f"Expected entity to be a string, got {type(self.entity)}"
        assert len(self.entity) > 0, "Entity name cannot be empty"
        assert type(self.entity_index) == int, f"Expected index to be an integer, got {type(self.entity_index)}"
        assert 0 <= self.entity_index < len(metadata_filtered), f"Index {self.entity_index} is out of bounds for metadata of length {len(metadata_filtered)}"

    def _serialize_parameters(self) -> str:
        if self.entity is None:
            self._resolve_entity()
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}_{self.entity}_{self.seed}"


    @classmethod
    def plot(cls, data: dict, figsize: Optional[Tuple[int, int]] = (18, 4), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        task = data['metadata']['task']
        unlearning_algorithm = data['metadata']['unlearning_algorithm']
        interference_pair = data['metadata']['interference_pair']

        entity = data['metadata']['entity']
        displayed_entities = data['result']['displayed_entities']
        is_worst_biggest = data['result']['is_worst_biggest']
        num_train_epochs = data['result']['num_train_epochs']
        seed = data['metadata']['seed']
        interference_values = data['result']['interference_values']

        fig, axes = plt.subplots(2, 9, figsize=figsize)
        plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.88)

        # Images
        for row, state in enumerate(['off', 'on']):  # off = base model (row 0), on = unlearned (row 1)
            for col, name in enumerate(displayed_entities):
                ax = axes[row, col]
                ax.axis('off')
                ax.imshow(plt.imread(_decode_image(data['result']['images'][state][name])))

                if row == 0:
                    ax.set_title(get_target_overwrite(task, unlearning_algorithm, name)[0] + f'\n{interference_values[name]:.2f}', rotation=0, fontsize=9, pad=2, loc='center')

        # vertical row labels (written upwards)
        # compute vertical center of a row using one axis
        def row_center(ax):
            pos = ax.get_position()
            return (pos.y0 + pos.y1) / 2

        # compute x position for the left vertical label automatically from the leftmost axis position
        left_pos = axes[0, 0].get_position()
        left_x = left_pos.x0 - 0.01  # small offset to place label left of images
        fig.text(left_x, row_center(axes[0, 0]), 'Original', rotation=90, va='center', ha='center', fontsize=12, weight="bold")
        fig.text(left_x, row_center(axes[1, 0]), 'Unlearned', rotation=90, va='center', ha='center', fontsize=12, weight="bold")

        # group labels: compute center positions for the three groups using axes positions
        # groups: target (col 0), worst (cols 1-4), best (cols 5-8)
        def col_center(fig, ax_left, ax_right):
            pos_left = ax_left.get_position()
            pos_right = ax_right.get_position()
            return (pos_left.x0 + pos_right.x1) / 2

        # place group labels slightly above the figure (use y>1 to match requested style)
        fig.text(col_center(fig, axes[0, 0], axes[0, 0]), 0.98, "Target", ha="center", va="bottom", fontsize=12, weight="bold")
        fig.text(col_center(fig, axes[0, 1], axes[0, 4]), 0.98, f"Worst interfered ({interference_pair} {'↑' if is_worst_biggest else '↓'})", ha="center", va="bottom", fontsize=12, weight="bold")
        fig.text(col_center(fig, axes[0, 5], axes[0, 8]), 0.98, f"Least interfered ({interference_pair} {'↓' if is_worst_biggest else '↑'})", ha="center", va="bottom", fontsize=12, weight="bold")

        # Draw 2 vertical bars separating these 3 groups
        top_y = 1.0
        bottom_y = axes[1, 0].get_position().y0 - 0.005

        # x for boundary between Target (col 0) and Worst (col 1)
        pos_a = axes[0, 0].get_position()
        pos_b = axes[0, 1].get_position()
        x_boundary_1 = (pos_a.x1 + pos_b.x0) / 2

        # x for boundary between Worst (col 1-4) and Best (col 5-8)
        pos_c = axes[0, 4].get_position()
        pos_d = axes[0, 5].get_position()
        x_boundary_2 = (pos_c.x1 + pos_d.x0) / 2

        # draw bars
        for x in (x_boundary_1, x_boundary_2):
            line = Line2D([x, x], [bottom_y, top_y], transform=fig.transFigure, color='gray', linewidth=1.5, zorder=20)
            fig.add_artist(line)

        #if save_path:
        #    plt.savefig(save_path)
        if return_fig:
            return fig, ax
        else:
            plt.show()


    def _compute_from_scratch(self):
        self._resolve_entity()
        num_train_epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]        
        is_worst_biggest=mp_to_direction[self.interference_pair]!='↑'


        interference_per_pair = get_interference_per_pair(self.task, self.entity_index, self.unlearning_algorithm, num_train_epochs)
        all_names = list(interference_per_pair.keys())
        metric_list = [(name, interference_per_pair[name][self.interference_pair]) for name in all_names]  # list of (name, metric)

        if is_worst_biggest:
            metric_sorted_worst_first = sorted(metric_list, key=lambda x: x[1], reverse=True)  # worst first (largest)
            metric_sorted_best_first = sorted(metric_list, key=lambda x: x[1])  # best first (smallest)
        else:
            metric_sorted_worst_first = sorted(metric_list, key=lambda x: x[1])  # worst first (smallest)
            metric_sorted_best_first = sorted(metric_list, key=lambda x: x[1], reverse=True)  # best first (largest)
        worst = [n for n, _ in metric_sorted_worst_first if n != self.entity][:4]  # take 4 worst excluding target
        best = [n for n, _ in metric_sorted_best_first if n != self.entity and n not in worst][:4]  # take 4 best excluding target and avoiding duplicates
        assert len(worst) == 4, f"Expected 4 worst interfered, got {len(worst)}"
        assert len(best) == 4, f"Expected 4 best interfered, got {len(best)}"

        displayed_entities = [self.entity, *worst, *best]
        interference_values = {name: interference_per_pair[name][self.interference_pair] for name in displayed_entities}

        # Embed imagesin the json itself
        images = {'off': {}, 'on': {}}
        for state in ['off', 'on']:
            for name in displayed_entities:
                img_path = os.path.join(
                    get_generated_dataset_folder(self.task, self.unlearning_algorithm, num_train_epochs, get_target_overwrite(self.task, self.unlearning_algorithm, self.entity)[0]),
                    get_generated_dataset_file(state, self.seed, f"An image of {get_target_overwrite(self.task, self.unlearning_algorithm, name)[0]}")  # type: ignore
                )
                images[state][name] = _encode_image_file(img_path, max_dim=self.images_max_dim)
                #print(f"Encoded image for {name} in state {state}, interference value: {interference_values[name]:.2f}, {self.images_max_dim}")

        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_pair': self.interference_pair,
                'entity': self.entity,
                'entity_index': self.entity_index,
                'seed': self.seed,
            },
            'result': {
                'displayed_entities': displayed_entities,
                'worst': worst,
                'best': best,
                'is_worst_biggest': is_worst_biggest,
                'num_train_epochs': num_train_epochs,
                'interference_values': interference_values,
                'images': images,
            },
        }
        return data



class ResultTemplateMatrix(ResultTemplate):
    # I wrote this class to reuse the graph logic, because both InterferenceMatrix and ImplicitAssociationTest return a matrix that can be visualized with heatmap
    # But maybe that add way too mcuh confusion, because keys have different names...
    metric_key_name: str

    @classmethod
    def plot_make_title(cls, data: dict) -> str:
        raise NotImplementedError()

    @classmethod
    def plot(cls, data: dict, figsize: Optional[Tuple[int, int]] = None, cmap: str ="viridis", title: str = "", xlabel: str = "Receiver entity", ylabel: str = "Emitter entity", return_fig: bool =False) -> Optional[Tuple[Figure, plt.Axes]]:
        df = pd.DataFrame(data['result'])
        df.set_index('emitter', inplace=True)

        if df.shape[0] != df.shape[1]:
            raise ValueError("DataFrame must be square (same number of rows and columns).")
        if not np.all(df.index == df.columns):
            raise ValueError("Index and columns must be the same")
        if not title:
            title = cls.plot_make_title(data)

        df2 = df.dropna()

        if figsize is None:
            base = max(4, df2.shape[0] * 0.35)
            figsize = (base, base)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            df2.values,
            cmap=cmap,
            aspect="equal",
            interpolation="nearest"
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        ax.set_xticks(np.arange(df2.shape[1]))
        ax.set_yticks(np.arange(df2.shape[0]))

        # Larger index fonts
        ax.set_xticklabels(
            df2.columns.to_list(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=9,
        )

        ax.set_yticklabels(
            df2.index.to_list(),
            fontsize=9,
        )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)

        plt.tight_layout(pad=0.8)
        if return_fig:
            return fig, ax
        else:
            plt.show()



class ResultTemplateInterferenceMatrix(ResultTemplateMatrix):
    """
    *MetricInterferencePerEntityPair* between each possible combination of two *entities*
    within a *task*.

    **Arguments:** `m`, `t`, `u`, `m_p`.
    **Result:** `|t| x |t|` real-valued tensor.
    **Interpretation:** qualitative; visual patterns may be spotted, especially when
    rearranging indices in a meaningful manner (for example, grouping professions
    together). Further quantitative values may be derived, such as the average value or
    the ratio between the diagonal-average value and the non-diagonal-average value.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    metric_key_name: str = 'interference_pair'

    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}"

    @classmethod
    def plot_make_title(cls, data: dict) -> str:
        rt_pretty = data['metadata']['RT'].replace('ResultTemplate', '')
        task_pretty = data['metadata']['task'].title()
        method_pretty = data['metadata']['unlearning_algorithm'].title()
        metric_pretty = f"{data['metadata'][data['metadata']['_metric_key_name']].replace('_', ' ').title()} ({data['metadata']['metric_direction']})"
        title = f"{rt_pretty}\nTask: {task_pretty}\nMethod: {method_pretty}\nMetric: {metric_pretty}"
        return title


    def _compute_from_scratch(self):
        metadata_filtered = get_metadata_filtered(self.task)
        labels = [e['name'] for e in metadata_filtered]
        num_train_epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]

        # df_aggregated_interference = store one MetricInterferencePerEntityPair (brisque_diff, clip_diff, rmse, or ssim)
        df_aggregated_interference = pd.DataFrame(columns=labels)
        for index in range(len(labels)):
            if not os.path.exists(get_interference_per_pair_path(self.task, index, self.unlearning_algorithm, num_train_epochs)):
                logger.warning(f'SKIP entity-pair analysis for task={self.task}, index={index}, method={self.unlearning_algorithm}, num_train_epochs={num_train_epochs}, do not exist yet')
                continue
            #logger.info(f'Analyzing entity-pairs for task={self.task}, index={index}, method={self.unlearning_algorithm}, num_train_epochs={num_train_epochs}...')
            interference_per_pair = get_interference_per_pair(self.task, index, self.unlearning_algorithm, num_train_epochs)
            emitter_name = metadata_filtered[index]['name']
            df_aggregated_interference.loc[emitter_name] = [interference_per_pair[l][self.interference_pair] for l in labels]
            #df_aggregated_interference_clip_diff.loc[emitter_name] = [interference_per_pair[l]['clip_diff'] for l in labels]
            #df_aggregated_interference_rmse.loc[emitter_name] = [interference_per_pair[l]['rmse'] for l in labels]
            #df_aggregated_interference_ssim.loc[emitter_name] = [interference_per_pair[l]['ssim'] for l in labels]
            assert list(interference_per_pair.keys()) == labels, "Labels don't match"

        df_aggregated_interference.index.name = "emitter"
        df_aggregated_interference = df_aggregated_interference.reset_index()

        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                self.metric_key_name: self.interference_pair,
                '_metric_key_name': self.metric_key_name,
                'metric_direction': mp_to_direction[self.interference_pair],
            },
            'result': df_aggregated_interference.to_dict(orient='records'),
        }
        return data


def jacc_metric_score(entity_1: str, entity_2: str, metadata_filtered: List[Dict[str, Any]], entity_col: str = 'name') -> float:
    """
    Jaccard similarity between two entities, based on their attributes.
    Each attribute (column) contributes between 0 and 1 to the similarity
    We do not know the types and ranges of the attributes beforehand.
    For each attribute, both values for the two entities must be non-NaN and of the same type, otherwise we ignore that attribute (contribution 0).
    The calculation for each attribute is as follows:
    * If the attribute is categorical (str or bool), the contribution is 1 if the two entities have the same value for that attribute, and 0 otherwise.
    * If the attribute is numerical, and both values are between 0 and 1, the contribution is 1 - abs(value_1 - value_2)
    * If the attribute is numerical, and both values are between 1 and 100, the contribution is 1 - abs(value_1 - value_2) / 100
    * else, the contribution is 0 (we do not know how to handle it, so we ignore it)
    """
    # Get the rows corresponding to the two entities
    row_1 = next((row for row in metadata_filtered if row[entity_col] == entity_1), None)
    row_2 = next((row for row in metadata_filtered if row[entity_col] == entity_2), None)
    if row_1 is None or row_2 is None:
        raise ValueError(f"Entities {entity_1} and/or {entity_2} not found in metadata")
    if set(row_1.keys()) != set(row_2.keys()):
        raise ValueError(f"Entities {entity_1} and {entity_2} must have the same attributes")
    
    # Calculate similarity for each attribute
    similarity = 0.0
    valid_attributes = 0
    for attr in row_1.keys():
        value_1 = row_1[attr]
        value_2 = row_2[attr]

        if pd.isna(value_1) or pd.isna(value_2) or type(value_1) != type(value_2):
            continue  # ignore this attribute

        if isinstance(value_1, (str, bool)):
            similarity += 1.0 if value_1 == value_2 else 0.0
            valid_attributes += 1
        elif isinstance(value_1, (int, float)):
            if 0 <= value_1 <= 1 and 0 <= value_2 <= 1:
                similarity += 1 - abs(value_1 - value_2)
                valid_attributes += 1
            elif 1 < value_1 <= 100 and 1 < value_2 <= 100:
                similarity += 1 - abs(value_1 - value_2) / 100
                valid_attributes += 1
            else:
                continue  # ignore this attribute
        else:
            continue  # ignore this attribute
    similarity = similarity / valid_attributes if valid_attributes > 0 else 0.0

    # Post checks
    assert valid_attributes > 0, f"Expected at least one valid attribute for entities {entity_1} and {entity_2}, got {valid_attributes}."
    assert type(similarity) == float
    assert 0 <= similarity <= 1
    return similarity

class ResultTemplateSimilarityMatrix(ResultTemplateMatrix):
    """
    *Similarities* between each possible combination of two *entities* within a *task*.
    * **Arguments**: $m, t, s$
    * **Result**: $|t| \times |t|$ real-valued tensor
    * **Interpretation**: qualitative; visual patterns may be spotted, similarly to *InterferenceMatrix*.
    """
    model: type_model = 'sd1.4'
    task: type_task = 'scenes'
    similarity_metric: type_s = 'clip'
    metric_key_name: str = 'similarity_metric'


    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{self.similarity_metric}"

    def _get_partial_path_local(self):
        return self._get_data_path_local() + '.partial'

    @classmethod
    def plot_make_title(cls, data: dict) -> str:
        rt_pretty = data['metadata']['RT'].replace('ResultTemplate', '')
        task_pretty = data['metadata']['task'].title()
        metric_pretty = f"{data['metadata'][data['metadata']['_metric_key_name']].replace('_', ' ').title()}"
        title = f"{rt_pretty}\nTask: {task_pretty}\nMetric: {metric_pretty}"
        return title


    def _compute_from_scratch(self) -> dict:
        metadata_filtered: List[Dict[str, Any]] = get_metadata_filtered(self.task)
        labels: List[str] = [e['name'] for e in metadata_filtered]

        if self.similarity_metric == 'clip':
            # see calculate_similarity_clip
            # Dont fotget to save only when save_outputs==true... or assert save_outputs
            # Given the current implementation of calculate_similarity_clip, we probably assert save_outputs
            # To keep compatible with as it was done before, it should save a json with `orient='records'` with the content of `data['result']`
            raise NotImplementedError(f"Similarity matrix not found locally or in Hugging Face Hub. Please compute it first with calculate_similarity_clip")

        elif self.similarity_metric == 'jacc':
            # Load partial
            # 100x100 matrix
            if os.path.exists(self._get_partial_path_local()) and not self.recompute_if_exists:
                df_similarities = pd.read_json(self._get_partial_path_local(), orient='records')
                df_similarities.set_index('emitter', inplace=True)
                assert df_similarities.index.to_list() == labels
            else:
                df_similarities = pd.DataFrame(index=labels, columns=labels)

            # Calculate
            for entity_emitter, row_emitter in df_similarities.iterrows():
                print(f'Analying similarities for entity_emitter={entity_emitter}')
                for entity_receiver in row_emitter.index:
                    if pd.isna(df_similarities.loc[entity_emitter, entity_receiver]):  # type: ignore
                        similarity: float = jacc_metric_score(entity_emitter, entity_receiver, metadata_filtered)
                        df_similarities.loc[entity_emitter, entity_receiver] = similarity

                # Save partial at the end of each row
                df_similarities.reset_index(names='emitter').to_json(self._get_partial_path_local(), orient='records')
        
        # Return to be saved in its final form
        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                self.metric_key_name: self.similarity_metric,
                '_metric_key_name': self.metric_key_name,
            },
            'result': df_similarities.reset_index(names='emitter').to_dict(orient='records'),
        }
        return data

 

rt_name_to_class = {
    "MetricMetricAlignment": ResultTemplateMetricMetricAlignment,
    "MetricSimilarityAlignment": ResultTemplateMetricSimilarityAlignment,
    "InterferenceMatrix": ResultTemplateInterferenceMatrix,
    "SimilarityMatrix": ResultTemplateSimilarityMatrix,
    "SignificantRelationshipNumerical": ResultTemplateSignificantRelationshipNumerical,
    "SignificantRelationshipCategorical": ResultTemplateSignificantRelationshipCategorical,
    "CountSignificantRelationship": ResultTemplateCountSignificantRelationship,
    "ImplicitAssociationTest": ResultTemplateImplicitAssociationTest,
    "MinimumCutInterference": ResultTemplateMinimumCutInterference,
    "UnlearningVisualSummary": ResultTemplateUnlearningVisualSummary,
    "InterferenceVisualSummary": ResultTemplateInterferenceVisualSummary,
}


rt_name_to_params = {
    "MetricMetricAlignment": ["model", "task", "unlearning_algorithm", "interference_entity_1", "interference_entity_2"],
    "MetricSimilarityAlignment": ["model", "task", "unlearning_algorithm", "interference_pair", "similarity_metric"],
    "InterferenceMatrix": ["model", "task", "unlearning_algorithm", "interference_pair"],
    "SimilarityMatrix": ["model", "task", "similarity_metric"],
    "SignificantRelationshipNumerical": ["model", "task", "unlearning_algorithm", "interference_entity", "attribute"],
    "SignificantRelationshipCategorical": ["model", "task", "unlearning_algorithm", "interference_entity", "attribute", "attribute_value"],
    "CountSignificantRelationship": ["model", "task", "unlearning_algorithm", "interference_entity_list", "attribute_list"],
    "ImplicitAssociationTest": ["model", "task", "unlearning_algorithm", "attribute_1", "attribute_2", "latent_embedding"],
    "MinimumCutInterference": ["model", "task", "unlearning_algorithm", "interference_pair", "entity_1", "entity_2"],
    "UnlearningVisualSummary": ["model", "task", "unlearning_algorithm"],
    "InterferenceVisualSummary": ["model", "task", "unlearning_algorithm", "interference_pair", "entity"],
}
