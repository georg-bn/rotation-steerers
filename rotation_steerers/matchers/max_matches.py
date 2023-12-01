import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
import numpy as np
from DeDoDe.utils import dual_softmax_matcher, to_pixel_coords, to_normalized_coords
from rotation_steerers.matchers.dual_softmax_matcher import DualSoftMaxMatcher

class MaxMatchesMatcher(nn.Module):
    def __init__(self, rots, *args, steerer=None, projector=torch.nn.Identity(), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rots = rots 
        self.steerer = steerer
        self.projector = projector

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None, 
              normalize = False, inv_temp = 1, threshold = 0.0,
              steerer = None, rot_est = False):
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
        best_num_matches = -1
        matches_A = matches_B = batch_inds = None
        best_rot = 0
        descriptions_B = self.projector(descriptions_B)
        for rot in self.rots:
            P = dual_softmax_matcher(
                self.projector(
                    steerer.steer_descriptions(descriptions_A, {"nbr_rotations": rot})
                ),
                descriptions_B,
                normalize = normalize, inv_temperature=inv_temp,
            )
            inds = torch.nonzero((P == P.max(dim=-1, keepdim = True).values) 
                            * (P == P.max(dim=-2, keepdim = True).values) * (P > threshold))
            batch_inds = inds[:,0]
            num_matches = len(batch_inds)
            if num_matches > best_num_matches:    
                matches_A = keypoints_A[batch_inds, inds[:,1]]
                matches_B = keypoints_B[batch_inds, inds[:,2]]
                best_num_matches = num_matches
                best_rot = rot
        if rot_est:
            return best_rot
        else:
            return matches_A, matches_B, batch_inds

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)
    
    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)


class SubsetMatcher(nn.Module):
    def __init__(self, rots, *args, steerer=None, projector=torch.nn.Identity(), subset_size=1000, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.steerer = steerer
        self.subset_size = subset_size

        self.rot_matcher = MaxMatchesMatcher(rots=rots, steerer=steerer, projector=projector)
        self.ordinary_matcher = DualSoftMaxMatcher(projector=projector)

    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None, 
              normalize = False, inv_temp = 1, threshold = 0.0):
        subsample_A = np.random.choice(keypoints_A.shape[-2],
                                       size=self.subset_size,
                                       replace=False)
        subsample_B = np.random.choice(keypoints_B.shape[-2],
                                       size=self.subset_size,
                                       replace=False)
        best_rot = self.rot_matcher.match(
            keypoints_A[..., subsample_A, :], descriptions_A[..., subsample_A, :], 
            keypoints_B[..., subsample_B, :], descriptions_B[..., subsample_B, :],
            P_A = None if P_A is None else P_A[..., subsample_A],
            P_B = None if P_B is None else P_B[..., subsample_B],
            normalize = normalize, inv_temp = inv_temp, threshold = threshold,
            rot_est=True,
        )

        descriptions_A = self.steerer.steer_descriptions(descriptions_A, {"nbr_rotations": best_rot})
        return self.ordinary_matcher.match(
            keypoints_A, descriptions_A,
            keypoints_B, descriptions_B,
            P_A=P_A, P_B=P_B,
            normalize=normalize, inv_temp=inv_temp, threshold=threshold,
        )

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)
    
    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)
