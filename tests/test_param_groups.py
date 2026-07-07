import torch
from torch import nn

from oge.train_utils import make_weight_decay_param_groups


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, bias=True)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 5, bias=True)
        self.ln = nn.LayerNorm(5)
        self.classifier = nn.Linear(5, 2, bias=True)


def names_in(group, model):
    ids = {id(p) for p in group["params"]}
    return {name for name, p in model.named_parameters() if id(p) in ids}


def test_weight_decay_param_groups_policy():
    model = ToyModel()
    groups = make_weight_decay_param_groups(model, weight_decay=0.01, policy="weights_only_no_bias_norm")
    decay_names = names_in(groups[0], model)
    no_decay_names = names_in(groups[1], model)

    assert groups[0]["weight_decay"] == 0.01
    assert groups[1]["weight_decay"] == 0.0
    assert {"conv.weight", "fc.weight", "classifier.weight"}.issubset(decay_names)
    assert {"conv.bias", "bn.weight", "bn.bias", "fc.bias", "ln.weight", "ln.bias", "classifier.bias"}.issubset(no_decay_names)
    assert decay_names.isdisjoint(no_decay_names)
