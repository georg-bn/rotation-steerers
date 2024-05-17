import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from DeDoDe.utils import *
import DeDoDe

class DescriptorLoss(nn.Module):
    
    def __init__(self,
                 detector,
                 num_keypoints = 5000,
                 normalize_descriptions = False,
                 inv_temp = 1,
                 device = get_best_device()) -> None:
        super().__init__()
        self.detector = detector
        self.tracked_metrics = {}
        self.num_keypoints = num_keypoints
        self.normalize_descriptions = normalize_descriptions
        self.inv_temp = inv_temp
    
    def warp_from_depth(self, batch, kpts_A, kpts_B):
        mask_A_to_B, kpts_A_to_B = warp_kpts(kpts_A, 
                    batch["im_A_depth"],
                    batch["im_B_depth"],
                    batch["T_1to2"],
                    batch["K1"],
                    batch["K2"],)
        mask_B_to_A, kpts_B_to_A = warp_kpts(kpts_B, 
                    batch["im_B_depth"],
                    batch["im_A_depth"],
                    batch["T_1to2"].inverse(),
                    batch["K2"],
                    batch["K1"],)
        return (mask_A_to_B, kpts_A_to_B), (mask_B_to_A, kpts_B_to_A)
    
    def warp_from_homog(self, batch, kpts_A, kpts_B):
        kpts_A_to_B = homog_transform(batch["Homog_A_to_B"], kpts_A)
        kpts_B_to_A = homog_transform(batch["Homog_A_to_B"].inverse(), kpts_B)
        return (None, kpts_A_to_B), (None, kpts_B_to_A)

    def supervised_loss(self,
                        outputs,
                        batch,
                        rot_A=0,
                        rot_B=0,
                        steerer=None,
                        continuous_rot=False,
                       ):
        kpts_A, kpts_B = self.detector.detect(batch, num_keypoints = self.num_keypoints)['keypoints'].clone().chunk(2)

        desc_grid_A, desc_grid_B = outputs["description_grid"].chunk(2)

        desc_A = F.grid_sample(desc_grid_A.float(), kpts_A[:,None], mode = "bilinear", align_corners = False)[:,:,0].mT
        desc_B = F.grid_sample(desc_grid_B.float(), kpts_B[:,None], mode = "bilinear", align_corners = False)[:,:,0].mT

        # rotate keypoints back so that GT annotations can be used
        kpts_A = kpts_A.clone()
        kpts_B = kpts_B.clone()
        if continuous_rot:
            cosA, sinA = np.cos(rot_A), np.sin(rot_A)
            cosB, sinB = np.cos(rot_B), np.sin(rot_B)
            R_A_transpose = torch.tensor([[cosA, sinA],
                                          [-sinA, cosA]],
                                         dtype=kpts_A.dtype,
                                         device=kpts_A.device)
            R_B_transpose = torch.tensor([[cosB, sinB],
                                          [-sinB, cosB]],
                                         dtype=kpts_B.dtype,
                                         device=kpts_B.device)
            kpts_A = kpts_A @ R_A_transpose
            kpts_B = kpts_B @ R_B_transpose
        else:
            if rot_A == 1:
                kpts_A[..., [0, 1]] = kpts_A[..., [1, 0]]
                kpts_A[..., 0] = -kpts_A[..., 0]
            elif rot_A == 2:
                kpts_A = -kpts_A
            elif rot_A == 3:
                kpts_A[..., [0, 1]] = kpts_A[..., [1, 0]]
                kpts_A[..., 1] = -kpts_A[..., 1]

            if rot_B == 1:
                kpts_B[..., [0, 1]] = kpts_B[..., [1, 0]]
                kpts_B[..., 0] = -kpts_B[..., 0]
            elif rot_B == 2:
                kpts_B = -kpts_B
            elif rot_B == 3:
                kpts_B[..., [0, 1]] = kpts_B[..., [1, 0]]
                kpts_B[..., 1] = -kpts_B[..., 1]

        if "im_A_depth" in batch:
            (mask_A_to_B, kpts_A_to_B), (mask_B_to_A, kpts_B_to_A) = self.warp_from_depth(batch, kpts_A, kpts_B)
        elif "Homog_A_to_B" in batch:
            (mask_A_to_B, kpts_A_to_B), (mask_B_to_A, kpts_B_to_A) = self.warp_from_homog(batch, kpts_A, kpts_B)
            
        if steerer is not None and (rot_A > 0 or rot_B > 0):
            if continuous_rot:
                rot = rot_A - rot_B
                if rot < 0:
                    rot = rot + 2 * np.pi
                desc_B = steerer(desc_B, rot)
            else:
                nbr_rot = (4 + rot_A - rot_B) % 4  # nbr of rotations to align B with A
                for _ in range(nbr_rot):
                    desc_B = steerer(desc_B)

        with torch.no_grad():
            D_B = torch.cdist(kpts_A_to_B, kpts_B)
            D_A = torch.cdist(kpts_A, kpts_B_to_A)
            inds = torch.nonzero((D_B == D_B.min(dim=-1, keepdim = True).values) 
                                 * (D_A == D_A.min(dim=-2, keepdim = True).values)
                                 * (D_B < 0.01)
                                 * (D_A < 0.01))

        logP_A_B = dual_log_softmax_matcher(desc_A, desc_B, 
                                            normalize = self.normalize_descriptions,
                                            inv_temperature = self.inv_temp)
        neg_log_likelihood = -logP_A_B[inds[:,0], inds[:,1], inds[:,2]].mean()

        self.tracked_metrics["neg_log_likelihood"] = (
            0.99 * self.tracked_metrics.get("neg_log_likelihood", neg_log_likelihood.detach().item())
            + 0.01 * neg_log_likelihood.detach().item()
        )
        if np.random.rand() > 0.99:
            print(f'nll: {self.tracked_metrics["neg_log_likelihood"]}')

        loss = neg_log_likelihood

        return loss
    
    def forward(self,
                outputs,
                batch,
                rot_A=0,
                rot_B=0,
                steerer=None,
                continuous_rot=False,
               ):
        losses = self.supervised_loss(outputs,
                                      batch,
                                      rot_A,
                                      rot_B,
                                      steerer=steerer,
                                      continuous_rot=continuous_rot,
                                     )
        return losses
