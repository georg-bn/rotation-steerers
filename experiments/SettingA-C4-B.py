import os
from argparse import ArgumentParser

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset
import torch.nn as nn

from DeDoDe.datasets.megadepth import MegadepthBuilder
from DeDoDe.encoder import VGG
from DeDoDe.decoder import ConvRefiner, Decoder
from DeDoDe import dedode_detector_L
from DeDoDe.benchmarks import MegadepthNLLBenchmark
from DeDoDe.model_zoo import dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G

from rotation_steerers.train import train_k_steps
from rotation_steerers.descriptor_loss import DescriptorLoss
from rotation_steerers.checkpoint import CheckPoint
from rotation_steerers.steerers import DiscreteSteerer


def train(detector_weights, descriptor):
    NUM_PROTOTYPES = 256 # == descriptor size
    model = descriptor.cuda()
    model.eval()

    generator = torch.nn.Linear(in_features=NUM_PROTOTYPES,
                                out_features=NUM_PROTOTYPES,
                                bias=False).cuda().weight.data

    steerer = DiscreteSteerer(generator)

    params = [
        {"params": steerer.parameters(), "lr": 1e-3},
    ]
    optim = AdamW(params, weight_decay = 0)
    n0, N, k = 0, 10_000, 1000
    lr_scheduler = CosineAnnealingLR(optim, T_max = N)
    import os
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    checkpointer = CheckPoint("workspace/", name = experiment_name, only_steerer=True)
    
    model, optim, lr_scheduler, n0 = checkpointer.load(model, optim, lr_scheduler, n0, steerer=steerer)

    detector = dedode_detector_L(weights = detector_weights)
    loss = DescriptorLoss(detector=detector, normalize_descriptions = True, inv_temp = 20)
    

    H, W = 512, 512
    mega = MegadepthBuilder(data_root="data/megadepth", loftr_ignore=True, imc21_ignore = True, use_detections=False)
    use_horizontal_flip_aug = False
    megadepth_train1 = mega.build_scenes(
        split="train_loftr", min_overlap=0.01, ht=H, wt=W, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug,
    )
    megadepth_train2 = mega.build_scenes(
        split="train_loftr", min_overlap=0.35, ht=H, wt=W, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug,
    )

    megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
    mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)
    
    megadepth_test = mega.build_scenes(
        split="test_loftr", min_overlap=0.01, ht=H, wt=W, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug,
    )
    mega_test = MegadepthNLLBenchmark(ConcatDataset(megadepth_test))
    # grad_scaler = torch.cuda.amp.GradScaler()
    
    checkpointer.save(model, optim, lr_scheduler, -1, steerer=steerer, label=-1)
    for n in range(n0, N, k):
        mega_sampler = torch.utils.data.WeightedRandomSampler(
            mega_ws, num_samples = 8 * k, replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                megadepth_train,
                batch_size = 8,
                sampler = mega_sampler,
                num_workers = 8,
            )
        )
        train_k_steps(
            n, k, mega_dataloader, model, loss, optim, lr_scheduler, grad_scaler = None, rot90=True, steerer=steerer,
        )
        checkpointer.save(model, optim, lr_scheduler, n, steerer=steerer, label=n)
        # mega_test.benchmark(detector = detector, descriptor = model)


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"

    parser = ArgumentParser()
    parser.add_argument("--detector_path", default="dedode_detector_C4.pth")
    parser.add_argument("--descriptor_path", default="dedode_descriptor_B.pth")
    args = parser.parse_args()
    weights = torch.load(args.detector_path)
    descriptor = dedode_descriptor_B(weights = torch.load(args.descriptor_path))

    train(detector_weights=weights, descriptor=descriptor)
