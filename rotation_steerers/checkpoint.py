import os
import torch
import gc

import DeDoDe

class CheckPoint:
    def __init__(self, dir=None, name="tmp", only_steerer=False):
        self.name = name
        self.dir = dir
        self.only_steerer = only_steerer
        os.makedirs(self.dir, exist_ok=True)

    def save(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
        steerer=None,
        label="latest",
        ):
        assert model is not None
        if steerer is not None:
            if self.only_steerer:
                states = {
                    "n": n,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "steerer": steerer.state_dict(),
                }
            else:
                states = {
                    "model": model.state_dict(),
                    "n": n,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "steerer": steerer.state_dict(),
                }
        else:
            states = {
                "model": model.state_dict(),
                "n": n,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
        torch.save(states, self.dir + self.name + f"_{label}.pth")
        print(f"Saved states {list(states.keys())}, at step {n}")
    
    def load(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
        steerer=None,
        ):
        if os.path.exists(self.dir + self.name + f"_latest.pth"):
            states = torch.load(self.dir + self.name + f"_latest.pth")
            if "model" in states:
                model.load_state_dict(states["model"])
            if "n" in states:
                n = states["n"] if states["n"] else n
            if "optimizer" in states:
                try:
                    optimizer.load_state_dict(states["optimizer"])
                except Exception as e:
                    print(f"Failed to load states for optimizer, with error {e}")
            if "lr_scheduler" in states:
                lr_scheduler.load_state_dict(states["lr_scheduler"])
            if steerer is not None and "steerer" in states:
                steerer.load_state_dict(states["steerer"])
            print(f"Loaded states {list(states.keys())}, at step {n}")
            del states
            gc.collect()
            torch.cuda.empty_cache()
        return model, optimizer, lr_scheduler, n
