import os
import sys
import torch
from functools import wraps
from urllib.request import urlopen
from shutil import copyfileobj

from EpiScan.models.embedding import FullyConnectedEmbed, SkipLSTM
from EpiScan.models.contact_sep import ContactCNN
from EpiScan.models.interaction_sep import ModelInteraction
from EpiScan.embLLM.utils import log

STATE_DICT_BASENAME = "dscript_{version}.pt"
MODEL_URL_BASE = "http://cb.csail.mit.edu/cb/dscript/data/models/"

def create_model(model_class, *args, **kwargs):
    return model_class(*args, **kwargs)

def load_model(model, state_dict_path):
    model.load_state_dict(torch.load(state_dict_path))
    return model.eval()

def build_lm_1(state_dict_path):
    return load_model(create_model(SkipLSTM, 21, 100, 1024, 3), state_dict_path)

def build_human_1(state_dict_path):
    embModel = create_model(FullyConnectedEmbed, 6165, 100, 0.5)
    conModel = create_model(ContactCNN, 100, 50, 7)
    model = create_model(ModelInteraction, embModel, conModel, use_cuda=True, do_w=True, do_pool=True, do_sigmoid=True, pool_size=9)
    return load_model(model, state_dict_path)

VALID_MODELS = {"lm_v1": build_lm_1, "human_v1": build_human_1}

def get_state_dict_path(version):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), STATE_DICT_BASENAME.format(version=version))

def download_state_dict(version, verbose=True):
    state_dict_path = get_state_dict_path(version)
    state_dict_url = f"{MODEL_URL_BASE}{STATE_DICT_BASENAME.format(version=version)}"
    
    if not os.path.exists(state_dict_path):
        try:
            if verbose:
                log(f"Downloading model {version} from {state_dict_url}...")
            with urlopen(state_dict_url) as response, open(state_dict_path, "wb") as out_file:
                copyfileobj(response, out_file)
        except Exception as e:
            log(f"Unable to download model - {e}")
            sys.exit(1)
    
    return state_dict_path

def retry_decorator(retry_count):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retry_count):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    version = args[0]
                    log(f"\033[93mLoading {version} from disk failed. Retrying download attempt: {attempt + 1}\033[0m")
                    if e.args[0].startswith("unexpected EOF"):
                        state_dict_path = get_state_dict_path(version)
                        if os.path.exists(state_dict_path):
                            os.remove(state_dict_path)
                    else:
                        raise e
            raise Exception(f"Failed to download {version}")
        return wrapper
    return decorator

@retry_decorator(3)
def get_pretrained(version="human_v1"):
    if version not in VALID_MODELS:
        raise ValueError(f"Model {version} does not exist")
    
    state_dict_path = download_state_dict(version)
    return VALID_MODELS[version](state_dict_path)
