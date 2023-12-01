import torch
import torch.nn.functional as F

class SteererDiscrete(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def steer_descriptions(self, descriptions, transformation_params, normalize = False):
        for _ in range(transformation_params["nbr_rotations"]):
            descriptions = self.model(descriptions)
        if normalize:
            descriptions = F.normalize(descriptions, dim = -1)
        return descriptions


if __name__ == "__main__":
    pass
