Steerers: A framework for rotation equivariant keypoint descriptors
---
TODO: paper link

A steerer is a linear map that modifies keypoint descriptions as if they were obtained from a rotated image. More info [below](#short-summary).

<img src="example_images/method.png" width="500">

See the example notebook [demo.ipynb](demo.ipynb) for simple matching examples.
Before running it, create a new virtual environment of your preference (e.g. conda) with `python>=3.9`, `jupyter notebook` and GPU-enabled PyTorch.
Then install the `rotation_steerers` package using pip (this automatically installs DeDoDe from GitHub as well, see `setup.py`): 
```
pip install .
```
The weights are uploaded to [releases](https://github.com/georg-bn/rotation-steerers/releases). To download model weights needed for the demo and put them in a new folder `model_weights`, run
```
bash download_weights.sh
```

We will publish training code and further model weights shortly.

## Short summary
A steerer is a linear map that modifies keypoint descriptions as if they were obtained from a rotated image.
So a steerer makes the keypoint descriptor equivariant.
This provides a computational shortcut as we don't have to rerun our descriptor for every rotation of the image.
Below we show matches from original DeDoDe descriptions, and DeDoDe descriptions multiplied by a steerer:

<img src="example_images/dedode_matches.png" width="700"> 

<img src="example_images/steered_dedode_matches.png" width="700">

Hence, steerers are useful practically. But they are also interesting conceptually: Steerers can be found for arbitrary given descriptors, even such that are trained without any rotation augmentation (we explain why in the paper). Further, steerers can be trained jointly with a descriptor, enabling rotation augmentation without degrading performance on upright images (we get interesting training dynamics, see [below](#evolution-of-eigenvalues-of-the-steerer-during-training) and Section 5.5 of the paper). Similarly we can fix a steerer and train a descriptor given the steerer. All these three settings are explored in our paper.

We consider steerers for the group C4 of quarter turn rotations as well as the full rotation group SO(2). The first case is useful practically to align images upright and the second for completely rotation invariant matching.

## Evolution of eigenvalues of the steerer during training

Here we provide the gif version of Figure 4 in the paper. We show how the eigenvalues of the steerer evolve in the complex plane during training when we train the steerer and descriptor jointly. The first three correspond to a C4-steerer and the last two show the Lie algebra generator of an SO(2)-steerer.

<img src="eigen_gifs/C4_standard_init.gif" width="250"><img src="eigen_gifs/C4_perm_init.gif" width="250"><img src="eigen_gifs/C4_inv_init.gif" width="250"><img src="eigen_gifs/SO2_standard_init.gif" width="250"><img src="eigen_gifs/SO2_spread_init.gif" width="250">

## Citation
```
TODO
```

## License
Our code has an MIT license. DeDoDe-G uses a DINOv2 backbone which has an Apache-2 license.
