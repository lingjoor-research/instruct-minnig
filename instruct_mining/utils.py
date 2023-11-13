import torch

# import: standard
from dotenv import load_dotenv
from os import getenv

# import: huggingface
from huggingface_hub import login


def get_device():
    """
    Returns the device to be used for training and inference.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def authenticate_hf() -> None:
    """
    Login huggingface with token.
    """
    
    load_dotenv()
    HF_TOKEN = getenv("HF_TOKEN")
    
    login(token=HF_TOKEN)
