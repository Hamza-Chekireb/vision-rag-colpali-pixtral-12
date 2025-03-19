#%%
# Colpali Model
from colpali_engine.models import ColPali

# Colpali queries and images preprocessing
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor

# Retruever Processor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

# Accelerate calculations
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device

# 
import torch

# Type Validation
from typing import List, cast
# %%
# 1. Embedding model importation
device = get_torch_device('cpu')
model_name = "vidore/colpali-v1.2"
model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

# %%
#2. Used to process queries and images to fit the model's input requirements beforehand
processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

# %%
