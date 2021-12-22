import torch
import torchaudio
import io
import os
import requests
import tarfile

import matplotlib.pyplot as plt
from IPython.display import Audio, display
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

#1. 데이터
