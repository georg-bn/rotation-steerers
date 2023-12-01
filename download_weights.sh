#!/usr/bin/env bash

mkdir -p model_weights

# Original DeDoDe weights
wget -P model_weights/ https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth
wget -P model_weights/ https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth

# Our weights
wget -P model_weights/ https://github.com/georg-bn/rotation-steerers/releases/download/release-1/B_steerer_setting_A.pth
wget -P model_weights/ https://github.com/georg-bn/rotation-steerers/releases/download/release-1/B_SO2_Spread_descriptor_setting_B.pth
wget -P model_weights/ https://github.com/georg-bn/rotation-steerers/releases/download/release-1/B_SO2_Spread_steerer_setting_B.pth
wget -P model_weights/ https://github.com/georg-bn/rotation-steerers/releases/download/release-1/B_C4_Perm_descriptor_setting_C.pth
wget -P model_weights/ https://github.com/georg-bn/rotation-steerers/releases/download/release-1/B_C4_Perm_steerer_setting_C.pth

exit 0
