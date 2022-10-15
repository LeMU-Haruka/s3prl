import torch

from s3prl.hub import my_model
from s3prl.upstream.example.model_config import load_config

ckpt = 'F:\OneDrive\pretrain_models\\fusion_3.pt'
model_config = load_config()
model = my_model(ckpt, model_config)
input = torch.rand(1, 100000)
feature = model([input])