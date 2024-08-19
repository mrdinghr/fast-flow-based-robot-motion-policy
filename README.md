# Flow matching policies

This repository contains the source code to pf my master thesis 'Fast Flow-based Robot Motion Policy'.



# Examples
## Euclidean & Sphere Push-T

For each example, train the model by running example_lightning.py. 
The model will be trained according to the parameters specified in the config example.yaml.
After training, the results can be visualized by running example_plots.py. 

The examples rfm_manifold_lasa train Riemannian flow matching on the lasa dataset, by considering the letter as a distribution.

The examples refcond_rfm_manifold_lasa train Riemannian flow matching policies by considering the letter as a trajectory. 
It uses one reference and one context observation in the form of previous states as conditioning, akin to "Efficient Video Prediction via Sparsely Conditioned Flow Matching".

The examples vision_refcond_rfm_manifold_lasa is as above but with observations defined as images.

## Diffusion policy

The examples were adapted from the diffusion policy repo (https://github.com/real-stanford/diffusion_policy/) with the aim to compare RFM and DP in various examples.
The examples can be run from examples/diffusion_pusht.
