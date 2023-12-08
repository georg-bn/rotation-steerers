import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DeDoDe.utils import to_pixel_coords, to_normalized_coords

class MaxSimilarityMatcher(nn.Module):        
    def __init__(self, steerer_order, steerer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.steerer_order = steerer_order 
        self.steerer = steerer

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None, 
              normalize = False, inv_temp = 1, threshold = 0.0):

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
        if normalize:
            descriptions_B = descriptions_B/descriptions_B.norm(dim=-1, keepdim=True)
        for power in range(self.steerer_order):
            if power > 0:
                descriptions_A = self.steerer.steer_descriptions(descriptions_A)
            if normalize:
                descriptions_A = descriptions_A/descriptions_A.norm(dim=-1, keepdim=True)
            if power == 0:
                corr = torch.einsum("b n c, b m c -> b n m",
                                    descriptions_A,
                                    descriptions_B) * inv_temp
            else:
                corr_rot = torch.einsum("b n c, b m c -> b n m",
                                        descriptions_A,
                                        descriptions_B) * inv_temp
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


class ContinuousMaxSimilarityMatcher(nn.Module):        
    def __init__(self, angles, *args, steerer=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # assumes no repeated entires in rots
        self.angles = angles 
        self.steerer = steerer

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None, 
              normalize = False, inv_temp = 1, threshold = 0.0):

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
        if normalize:
            descriptions_B = descriptions_B/descriptions_B.norm(dim=-1, keepdim=True)
        for angle in self.angles:
            descriptions_A_rot = self.steerer.steer_descriptions(descriptions_A, angle)
            if normalize:
                descriptions_A_rot = descriptions_A_rot/descriptions_A_rot.norm(dim=-1, keepdim=True)
            if angle == self.angles[0]:
                corr = torch.einsum("b n c, b m c -> b n m",
                                    descriptions_A_rot,
                                    descriptions_B) * inv_temp
            else:
                corr_rot = torch.einsum("b n c, b m c -> b n m",
                                        descriptions_A_rot,
                                        descriptions_B) * inv_temp
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
