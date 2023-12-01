import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
import numpy as np
from DeDoDe.utils import to_pixel_coords, to_normalized_coords

class MaxSimilarityMatcher(nn.Module):        
    def __init__(self, rots, *args, steerer=None, projector=torch.nn.Identity(), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # assumes no repeated entires in rots
        self.rots = rots 
        self.steerer = steerer
        self.projector = projector

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None, 
              normalize = False, inv_temp = 1, threshold = 0.0, steerer = None):
        if steerer is None:
            steerer = self.steerer

        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                               inv_temp = inv_temp, threshold = threshold) 
                    for k_A,d_A,k_B,d_B in
                    zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds

        matches_A = matches_B = batch_inds = None
        descriptions_B = self.projector(descriptions_B)
        if normalize:
            descriptions_B = descriptions_B/descriptions_B.norm(dim=-1, keepdim=True)
        for rot in self.rots:
            descriptions_A_rot = self.projector(
                steerer.steer_descriptions(descriptions_A, {"nbr_rotations": rot})
            )
            if normalize:
                descriptions_A_rot = descriptions_A_rot/descriptions_A_rot.norm(dim=-1, keepdim=True)
            if rot == self.rots[0]:
                corr = torch.einsum("b n c, b m c -> b n m", descriptions_A_rot, descriptions_B) * inv_temp
            else:
                corr_rot = torch.einsum("b n c, b m c -> b n m", descriptions_A_rot, descriptions_B) * inv_temp
                corr = torch.maximum(corr, corr_rot)


        P = corr.softmax(dim = -2) * corr.softmax(dim= -1)
        inds = torch.nonzero((P == P.max(dim=-1, keepdim = True).values) 
                        * (P == P.max(dim=-2, keepdim = True).values) * (P > threshold))
        batch_inds = inds[:,0]
        matches_A = keypoints_A[batch_inds, inds[:,1]]
        matches_B = keypoints_B[batch_inds, inds[:,2]]
        return matches_A, matches_B, batch_inds

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)
    
    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)
