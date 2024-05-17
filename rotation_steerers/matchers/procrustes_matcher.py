import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
import numpy as np
from DeDoDe.utils import to_pixel_coords, to_normalized_coords

def closest_rot_2x2(M, return_angle=False):
    # https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
    # (assumes positive determinant of input matrix)
    # NOT memory efficient
    M00 = M[..., 0, 0]
    M01 = M[..., 0, 1]
    M10 = M[..., 1, 0]
    M11 = M[..., 1, 1]
    return closest_rot_2x2_helper(M00, M01, M10, M11, return_angle=return_angle)

def closest_rot_2x2_helper(M00, M01, M10, M11, return_angle=False):
    E = 0.5 * (M00 + M11)
    H = 0.5 * (M10 - M01)
    if return_angle:
        return torch.atan2(H, E)
    hypothenuse = torch.sqrt(H**2 + E**2)
    cosP = E / hypothenuse
    sinP = H / hypothenuse
    return torch.stack([
        torch.stack([cosP, sinP], dim=-1),
        torch.stack([-sinP, cosP], dim=-1),
    ], dim=-1)

def procrustes_dual_softmax_matcher(desc_A: tuple['B','N','C'], desc_B: tuple['B','M','C'], inv_temperature = 1, normalize = False):
    if len(desc_A.shape) < 3:
        desc_A, desc_B = desc_A[None], desc_B[None]
    if normalize:
        desc_A = desc_A/desc_A.norm(dim=-1,keepdim=True)
        desc_B = desc_B/desc_B.norm(dim=-1,keepdim=True)

    B, N, C = desc_A.shape
    desc_A = desc_A.view(B, N, C//2, 2)
    BB, NB, CB = desc_B.shape
    desc_B = desc_B.view(BB, NB, CB//2, 2)

    # find optimal rotation from A to B and compute correlation there
    corr = torch.einsum("b n c u, b m c v -> b n m u v", desc_A, desc_B)
    ATB = closest_rot_2x2(corr)
    corr = torch.einsum("b n m u v, b n m u v -> b n m", corr, ATB) * inv_temperature
    P = corr.softmax(dim = -2) * corr.softmax(dim = -1)
    return P

class ProcrustesMatcher(nn.Module):        
    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None, 
              normalize = False, inv_temp = 1, threshold = 0.0,
             ):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                               inv_temp = inv_temp, threshold = threshold) 
                    for k_A,d_A,k_B,d_B in
                    zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds
        
        P = procrustes_dual_softmax_matcher(
            descriptions_A, descriptions_B, 
            normalize = normalize, inv_temperature=inv_temp,
        )
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

if __name__ == "__main__":
    pass
