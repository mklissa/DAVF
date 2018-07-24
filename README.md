# Diffusion-Based Approximate Value Functions
This repository contains the code for the [work](https://openreview.net/forum?id=BkgkoToZZ7) on Diffusion-Based Approximate Value Functions. It contains the implementations for the experiments on SparseMountainCar-v0 and SparseHalfCheetah-v0. To use these environments, you must first install [them](https://github.com/mklissa/my-gym).

# SparseMountainCar-v0

To reproduce the experiments on SparseMountainCar-v0, you should simply proceed by installing the requirements in any environment:

```
python setup.py install
```

After which, you can run the experiments:

```
cd SparseMountainCar
python main.py --gen_graph
```

# SparseHalfCheetah-v0

To reproduce the experiments on SparseHalfCheetah-v0, you first need to install OpenAI's [Baselines](https://github.com/openai/baselines). After which you should install the supplementary dependencies in the same environment as the one you installed the Baselines's dependencies:


```
. your_virtual_env/bin/activate
python setup.py install
```

After which, you can run the experiments:

```
cd ppo_gcn
python run_mujoco.py
```
