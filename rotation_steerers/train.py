import torch
import torchvision.transforms.functional as TTF
from tqdm import tqdm
import random
import numpy as np
from DeDoDe.utils import to_cuda, to_best_device


def train_step(train_batch,
               model,
               objective,
               optimizer,
               grad_scaler=None,
               rot90=False,
               rot_cont=False,
               steerer=None,
               **kwargs):
    optimizer.zero_grad()
    if rot90:
        nbr_rot_A = random.randint(0, 3)
        nbr_rot_B = random.randint(0, 3)
        train_batch["im_A"] = train_batch["im_A"].rot90(k=nbr_rot_A, dims=[-2, -1])
        train_batch["im_B"] = train_batch["im_B"].rot90(k=nbr_rot_B, dims=[-2, -1])
        out = model(train_batch)
        l = objective(out,
                      train_batch,
                      nbr_rot_A,
                      nbr_rot_B,
                      steerer=steerer)
    elif rot_cont:
        rot_A = 2 * np.pi * random.random()
        rot_B = 2 * np.pi * random.random()
        train_batch["im_A"] = TTF.rotate(
            train_batch["im_A"],
            np.rad2deg(rot_A),
            interpolation=TTF.InterpolationMode.BILINEAR,
        )
        train_batch["im_B"] = TTF.rotate(
            train_batch["im_B"],
            np.rad2deg(rot_B),
            interpolation=TTF.InterpolationMode.BILINEAR,
        )
        out = model(train_batch)
        l = objective(out,
                      train_batch,
                      rot_A,
                      rot_B,
                      steerer=steerer,
                      continuous_rot=True)
    else:
        out = model(train_batch)
        l = objective(out,
                      train_batch)
    if grad_scaler is not None:
        grad_scaler.scale(l).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        l.backward()
        optimizer.step()
    return {"train_out": out, "train_loss": l.item()}


def train_k_steps(n_0,
                  k,
                  dataloader,
                  model,
                  objective,
                  optimizer,
                  lr_scheduler,
                  grad_scaler = None,
                  rot90=False,
                  steerer=None,
                  rot_cont=False,
                  progress_bar=True):
    if rot_cont and rot90:
        raise ValueError()
    for n in tqdm(range(n_0, n_0 + k), disable=not progress_bar, mininterval = 10.):
        batch = next(dataloader)
        model.train(True)
        batch = to_best_device(batch)
        train_step(
            train_batch=batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            n=n,
            grad_scaler = grad_scaler,
            rot90 = rot90,
            steerer = steerer,
            rot_cont = rot_cont,
        )
        lr_scheduler.step()


def train_epoch(
    dataloader=None,
    model=None,
    objective=None,
    optimizer=None,
    lr_scheduler=None,
    epoch=None,
):
    model.train(True)
    print(f"At epoch {epoch}")
    for batch in tqdm(dataloader, mininterval=5.0):
        batch = to_best_device(batch)
        train_step(
            train_batch=batch, model=model, objective=objective, optimizer=optimizer
        )
    lr_scheduler.step()
    return {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "epoch": epoch,
    }


def train_k_epochs(
    start_epoch, end_epoch, dataloader, model, objective, optimizer, lr_scheduler
):
    for epoch in range(start_epoch, end_epoch + 1):
        train_epoch(
            dataloader=dataloader,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
        )
