import os
from argparse import ArgumentParser

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset
import torch.nn as nn

from DeDoDe.datasets.megadepth import MegadepthBuilder
from DeDoDe.descriptors.dedode_descriptor import DeDoDeDescriptor
from DeDoDe.encoder import VGG
from DeDoDe.decoder import ConvRefiner, Decoder
from DeDoDe import dedode_detector_L
from DeDoDe.benchmarks import MegadepthNLLBenchmark

from rotation_steerers.train import train_k_steps
from rotation_steerers.descriptor_loss import DescriptorLoss
from rotation_steerers.checkpoint import CheckPoint
from rotation_steerers.steerers import ContinuousSteerer


def train(detector_weights):
    NUM_PROTOTYPES = 256 # == descriptor size
    residual = True
    hidden_blocks = 5
    amp_dtype = torch.float16#torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,
            ),
            "4": ConvRefiner(
                256+256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,

            ),
            "2": ConvRefiner(
                128+128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,

            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks = hidden_blocks,
                residual = residual,
                amp = amp,
                amp_dtype = amp_dtype,
            ),
        }
    )
    import os
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    encoder = VGG(size = "19", pretrained = True, amp = amp, amp_dtype = amp_dtype)
    decoder = Decoder(conv_refiner, num_prototypes=NUM_PROTOTYPES)
    model = DeDoDeDescriptor(encoder = encoder, decoder = decoder).cuda()

    generator = torch.block_diag(
        *(
            torch.tensor([[0., 1],
                          [-1, 0]],
                         device='cuda')
            for _ in range(NUM_PROTOTYPES // 2)
        ),
    )
    steerer = ContinuousSteerer(generator)
    for param in steerer.parameters():
        param.requires_grad = False

    params = [
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.decoder.parameters(), "lr": 2e-4},
    ]
    optim = AdamW(params, weight_decay = 1e-5)
    n0, N, k = 0, 100_000, 1000
    lr_scheduler = CosineAnnealingLR(optim, T_max = N)
    checkpointer = CheckPoint("workspace/", name = experiment_name)
    
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
    grad_scaler = torch.cuda.amp.GradScaler()
    
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
            n, k, mega_dataloader, model, loss, optim, lr_scheduler, grad_scaler = grad_scaler, rot_cont=True, steerer=steerer,
        )
        checkpointer.save(model, optim, lr_scheduler, n, steerer=steerer)
        mega_test.benchmark(detector = detector, descriptor = model)


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    parser = ArgumentParser()
    parser.add_argument("--detector_path", default="dedode_detector_SO2.pth")
    args = parser.parse_args()
    weights = torch.load(args.detector_path)

    train(weights)
