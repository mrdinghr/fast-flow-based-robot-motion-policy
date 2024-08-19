# Fast Flow-based Robot Motion Policy

This repository contains the source code of my master thesis 'Fast Flow-based Robot Motion Policy'.

# Abstract
Diffusion policy (DP) shows good performance on Behavior Cloning and has ability to learn multi-modal action distribution, but get limits to long inference time and needs 100 Number of Function Evaluation (NFE) with DDPM and 10 NFE with DDIM. Consistency Policy proposes to learn a student policy from DP to shorten inference time but increases the difficulty and time of training. In this thesis, I implement Riemannian flow matching for robot motion policy generation (RFMP), which needs fewer NFE and is easier to train. FM has faster generation and training than diffusion model. I also propose stable riemannian flow matching (SRFMP), which generates stable result on Riemannian manifold. By 8 tasks in simulator and real world, I show that RFMP is more robust to NFE than DP, reachs 30Hz real-time control for real world robot and needs fewer training epochs than DP.\\


# Examples
'examples': train & test code for different policies (DP, RFMP, SRFMP) on 8 tasks.
6 tasks in simulator: Euclidean & Sphere Push-T, Robomimic Lift, Can, Square, Tool Hang with both state & vision based observation
2 tasks on real robot: Pick & Place, Rotate Mug
## Simulator: Euclidean & Sphere Push-T
<img src='media/pusht/pushT.gif' alt='drawing' width='100%'/>
<img src='media/pusht/sp_pusht.gif' alt='drawing' width='100%'/>

