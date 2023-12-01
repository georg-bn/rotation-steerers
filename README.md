Steerers: A framework for rotation equivariant keypoint descripors
---
TODO: paper link

See the example notebook `demo.ipynb` for a simple matching example.
Before running it, create a new virtual environment of your preference (e.g. conda) with `python>=3.9`, `jupyter notebook` and GPU-enabled PyTorch.
Then install the `rotation_steerer` package using pip (this automatically installs DeDoDe from GitHub as well, see `setup.py`): 
```
pip install .
```
To download model weights needed for the demo and put them in a new folder `model_weights`, run
```
bash download_weights.sh
```

We will publish training code and further model weights shortly.

## Citation
```
TODO
```

## License
Our code has an MIT license. DeDoDe-G uses a DINOv2 backbone which has an Apache-2 license.
