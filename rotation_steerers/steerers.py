import torch
import torch.nn.functional as F


class DiscreteSteerer(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = torch.nn.Parameter(generator)

    def forward(self, x):
        return F.linear(x, self.generator)

    def steer_descriptions(self, descriptions, steerer_power=1, normalize=False):
        for _ in range(steerer_power):
            descriptions = self.forward(descriptions)
        if normalize:
            descriptions = F.normalize(descriptions, dim=-1)
        return descriptions


class ContinuousSteerer(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = torch.nn.Parameter(generator)

    def forward(self, x, angle_radians):
        return F.linear(x, torch.matrix_exp(angle_radians * self.generator))

    def steer_descriptions(self, descriptions, angle_radians, normalize=False):
        descriptions = self.forward(descriptions, angle_radians)
        if normalize:
            descriptions = F.normalize(descriptions, dim=-1)
        return descriptions


if __name__ == "__main__":
    pass
