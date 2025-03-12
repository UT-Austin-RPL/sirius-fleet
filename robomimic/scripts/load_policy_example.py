import torch
from robomimic.utils.file_utils import policy_from_checkpoint
import sys

checkpoint = sys.argv[1]

model = policy_from_checkpoint(ckpt_path=checkpoint)[0].policy

