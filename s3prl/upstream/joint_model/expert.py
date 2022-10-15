from collections import OrderedDict
from typing import List, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .model_config import load_config
from .model_module import JointModel

HIDDEN_DIM = 8


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        self.name = "joint_model"

        print(
            f"{self.name} - You can use model_config to construct your customized model: {model_config}"
        )
        print(f"{self.name} - You can use ckpt to load your pretrained weights: {ckpt}")
        print(
            f"{self.name} - If you store the pretrained weights and model config in a single file, "
            "you can just choose one argument (ckpt or model_config) to pass. It's up to you!"
        )
        model_config = load_config()
        param = torch.load(ckpt)
        self.model = JointModel(model_config)
        self.model.encoder.load_state_dict(param)
        # The model needs to be a nn.Module for finetuning, not required for representation extraction
        # self.model1 = nn.Linear(1, HIDDEN_DIM)
        # self.model2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 1

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """
        hiddens = self.model.encode(wavs)

        padded_feats = pad_sequence(hiddens, batch_first=True)
        return {
            "last_hidden_state": padded_feats,
            "hidden_states": [padded_feats],
        }
