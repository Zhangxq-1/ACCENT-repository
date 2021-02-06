from main import _test
import torch


model_path=' '
model=torch.load(model_path)

_test(model)