# FACMAC: Factored Multi-Agent Centralised Policy Gradients (NeurIPS 2021)

This repo contains the code that was used in the paper "[FACMAC: Factored Multi-Agent Centralised Policy Gradients](https://arxiv.org/abs/2003.06709)".
It includes implementations for FACMAC, FACMAC-nonmonotonic, MADDPG, COMIX, COVDN, and Independent DDPG.

This codebase is built on top of the [PyMARL](https://github.com/oxwhirl/pymarl) framework for multi-agent reinforcement learning algorithms.
Please refer to that repo for more documentation.

## Setup instructions

Build the Dockerfile using
```
cd docker
bash build.sh
```
Note that the docker file here assumes that you have a licence key to install MuJoCo (otherwise you might encounter an error when building the docker image).

Set up StarCraft II and SMAC:
```
bash install_sc2.sh
```

## Environments

### Continuous Predator-Prey
We developed a simple variant of the simple tag environment from [Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).

To obtain a purely cooperative environment, we replace the prey's policy by a hard-coded heuristic, that, at any time step,
moves the prey to the sampled position with the largest distance to the closest predator. If one of the cooperative agents
collides with the prey, a team reward of +10 is given; otherwise, no reward is given.
For more details about the environment, please see the Appendix of the paper.

### Multi-Agent MuJoCo
We developed Multi-Agent MuJoCo (MAMuJoCo), a novel benchmark for continuous cooperative multi-agent robotic control.
Based on the popular fully observable single-agent robotic [MuJoCo](https://github.com/openai/mujoco-py) control suite from OpenAI Gym,
MAMuJoCo includes a wide variety of robotic control tasks in which multiple agents within a single robot have to solve a task cooperatively.
For more details about this benchmark, please check our [Multi-Agent MuJoCo](https://github.com/schroederdewitt/multiagent_mujoco) repo.

### StarCraft Multi-Agent Challenge (SMAC)
We also use the SMAC environment developed by [WhiRL](https://whirl.cs.ox.ac.uk/). Please check the [SMAC](https://github.com/oxwhirl/smac) repo for more details about the environment.
Note that for all SMAC experiments we used SC2.4.10 (not SC2.4.6.2.69232).
The results reported in the SMAC paper (https://arxiv.org/abs/1902.04043) use SC2.4.6.2.69232.
Performance is **not** always comparable across versions.

## Run an experiment 

Run an ALGORITHM from the folder `src/config/algs`
in an ENVIRONMENT from the folder `src/config/envs`
on a specific GPU using some PARAMETERS:
```
bash run.sh <GPU> python3 src/main.py --config=<ALGORITHM> --env-config=<ENVIRONMENT> with <PARAMETERS>
```

For example, to run the FACMAC algorithm on our continuous predator-prey task (with 3 agents) for 2mil timesteps using docker:
```
bash run.sh <GPU> python3 src/main.py --config=facmac_pp --env-config=particle with env_args.scenario_name=continuous_pred_prey_3a t_max=2000000
```

Another example, to run the FACMAC algorithm on some SMAC map (say '2s3z') for 2mil timesteps using docker:
```
bash run.sh <GPU> python3 src/main.py --config=facmac_smac --env-config=sc2 with env_args.map_name=2s3z t_max=2000000
```

The config files (src/config/algs/*.yaml) contain default hyper-parameters for the respective algorithms.
These were sometimes changed when running the experiments on different tasks.
Please see the Appendix of the paper for the exact hyper-parameters used.

For each environment you can specify the specific scenario by
including the parameter `with env_args.scenario_name=<scenario_name>` (for Continuous Predator-Prey and MAMuJoCo)
or `with env_args.map_name=<map_name>` (for SMAC).
For MAMuJoCo, please check the [Multi-Agent MuJoCo](https://github.com/schroederdewitt/multiagent_mujoco) repo for more details about different configuration parameters for the environment.


## Citing
If you used this code in your research or found it helpful, please consider citing the following paper:

Bibtex:
```
@inproceedings{peng2021facmac,
  title={FACMAC: Factored Multi-Agent Centralised Policy Gradients},
  author={Peng, Bei and Rashid, Tabish and de Witt, Christian Schroeder and Kamienny, Pierre-Alexandre and Torr, Philip and B{\"o}hmer, Wendelin and Whiteson, Shimon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```