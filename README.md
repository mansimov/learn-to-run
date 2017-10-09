# Learn to Run

This is OpenAI baselines repo somewhat adapted to run in python2.7

Follow instructions to install opensim-rl

```
https://github.com/stanfordnmbl/osim-rl
```

Make sure to look at below link in case of errors

```
https://github.com/stanfordnmbl/osim-rl#frequently-asked-questions
```

Fixes to memory leaking in RunEnv

```
https://github.com/stanfordnmbl/osim-rl/issues/58
```

Regular version of PPO
```
CUDA_VISIBLE_DEVICES=0 python baselines/ppo1/run_mujoco.py --seed 41 --env Walker2d-v1
```

VecEnv version of PPO (might still have bugs)
```
CUDA_VISIBLE_DEVICES=0 python baselines/ppo1/run_mujoco_vecenv.py --seed 41 --env Walker2d-v1
```

Note: Humanoid doesn't work with default PPO hyperparams use the following ones for now

```
pposgd_simple.learn(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_batch=4096,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=512,
        gamma=0.99, lam=0.95, schedule='adapt', desired_kl=0.02,
    )
```

LearnToRun with PPO (experimental)
```
CUDA_VISIBLE_DEVICES=0 python baselines/ppo1/run_mujoco_vecenv.py --seed 41 --env learntorun
```
