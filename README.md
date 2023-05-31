# Baselines

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones.

## Subpackages

- [A2C](baselines/a2c)
- [ACER](baselines/acer)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [GAIL](baselines/gail)
- [HER](baselines/her)
- [PPO1](baselines/ppo1) (obsolete version, left here temporarily)
- [PPO2](baselines/ppo2)
- [TRPO](baselines/trpo_mpi)

## Prerequisites

### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

### Mac OS X

Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:

```bash
brew install cmake openmpi
```

## Virtual environment

From the general python package sanity perspective, it is a good idea to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other. You can install virtualenv (which is itself a pip package) via

```bash
pip install virtualenv
```

Virtualenvs are essentially folders that have copies of python executable and all python packages.
To create a virtualenv called venv with python3, one runs:

```bash
virtualenv /path/to/venv --python=python3
```

To activate a virtualenv:

```bash
. /path/to/venv/bin/activate
```

More thorough tutorial on virtualenvs and options can be found [here](https://virtualenv.pypa.io/en/stable/)

## Python versions

Recommended Python version is 3.7.15

## Tensorflow versions

The master branch supports Tensorflow from version 1.4 to 1.14. For Tensorflow 2.0 support, please use tf2 branch.

## Installation

- Clone the repo and cd into it:

    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```

- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, you may use

    ```bash
    pip install tensorflow-gpu==1.14 # if you have a CUDA-compatible gpu and proper drivers
    pip install cudatoolkit=10.0.130 cudnn=7.6.5
    ```

    or

    ```bash
    pip install tensorflow==1.14
    ```

    to install Tensorflow 1.14, which is the latest version of Tensorflow supported by the master branch. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details.

- Install baselines package

    ```bash
    pip install -e .
    pip install matplotlib pandas gym[atari] filelock
    conda install ffmpeg
    ```

- Install atari Roms from [here](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) and extract the .rar file. After that, run:

    ```bash
    python -m atari_py.import_roms <path to folder>
    ```

## Testing the installation

All unit tests in baselines can be run using pytest runner:

```bash
pip install pytest
pytest
```

## Training models

Most of the algorithms in baselines repo are used as follows:

```bash
python -m baselines.run --alg=<name of the algorithm> --env=<environment_id> [additional arguments]
```

### Example 1. PPO with MuJoCo Humanoid

For instance, to train a fully-connected network controlling MuJoCo humanoid using PPO2 for 20M timesteps

```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7
```

Note that for mujoco environments fully-connected network is default, so we can omit `--network=mlp`
The hyperparameters for both network and the learning algorithm can be controlled via the command line, for instance:

```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7 --ent_coef=0.1 --num_hidden=32 --num_layers=3 --value_network=copy
```

will set entropy coefficient to 0.1, and construct fully connected network with 3 layers with 32 hidden units in each, and create a separate network for value function estimation (so that its parameters are not shared with the policy network, but the structure is the same)

See docstrings in [common/models.py](baselines/common/models.py) for description of network parameters for each type of model, and
docstring for [baselines/ppo2/ppo2.py/learn()](baselines/ppo2/ppo2.py#L152) for the description of the ppo2 hyperparameters.

### Example 2. DQN on Atari

DQN with Atari is at this point a classics of benchmarks. To run the baselines implementation of DQN on Atari Pong:

```bash
python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --num_timesteps=1e6
```

## Saving, loading and visualizing models

### Saving and loading the model

The algorithms serialization API is not properly unified yet; however, there is a simple method to save / restore trained models.
`--save_path` and `--load_path` command-line option loads the tensorflow state from a given path before training, and saves it after the training, respectively.
Let's imagine you'd like to train ppo2 on Atari Pong,  save the model and then later visualize what has it learnt.

```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=2e7 --save_path=~/models/pong_20M_ppo2
```

This should get to the mean reward per episode about 20. To load and visualize the model, we'll do the following - load the model, train it for 0 steps, and then visualize:

```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=0 --load_path=~/models/pong_20M_ppo2 --play
```

### Logging and vizualizing learning curves and other training metrics

By default, all summary data, including progress, standard output, is saved to a unique directory in a temp folder, specified by a call to Python's [tempfile.gettempdir()](https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir).
The directory can be changed with the `--log_path` command-line option.

```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=2e7 --save_path=~/models/pong_20M_ppo2 --log_path=~/logs/Pong/
```

*NOTE:* Please be aware that the logger will overwrite files of the same name in an existing directory, thus it's recommended that folder names be given a unique timestamp to prevent overwritten logs.

Another way the temp directory can be changed is through the use of the `$OPENAI_LOGDIR` environment variable.

For examples on how to load and display the training data, see [here](docs/viz/viz.ipynb).

## Trace Code

- main structure

    ```bash
    baselines/run.py
    main()->train()
        baselines/ppo2/ppo2.py
        learn()
            baselines/ppo2/model.py
            Model()
    ```

- common arguments (defined [here](baselines/common/cmd_util.py))
    1. env: environment ID, default='Reacher-v2'
    2. env_type: type of environment, used when the environment type cannot be automatically determined
    3. seed: RNG seed, default=None
    4. alg: Algorithm, default='ppo2'
    5. num_timesteps: default=1e6
    6. network: policy network type (mlp, cnn, lstm, cnn_lstm, conv_only), default=None
    7. gamestate: game state to load (so far only used in retro games)
    8. num_env: Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco, default=None
    9. reward_scale: Reward scale factor, default=1.0
    10. save_path: Path to save trained model to, default=None
    11. save_video_interval: Save video every x steps (0 = disabled), default=0
    12. save_video_length: Length of recorded video, default=200
    13. log_path: Directory to save learning curve data, default=None
    14. play: flag for visualization, default=False

- default policy network is set to cnn in [baselines/run.get_default_network()](baselines/run.py#L148). Inorder to share layers with value network, it should be set to conv_only, and then apply a fc layer to the latent.
- value network is specified in [baselines/common/policies.PolicyWithValue()](baselines/common/policies.py#L58) To share layers with policy network, set value_network='shared' for function build_policy(), defined in [baselines/common/policies.py](baselines/common/policies.py#L121), and used in [baselines/ppo2/ppo2.py](baselines/ppo2/ppo2.py#L88)
- POME adds two additional network, which are reward network and transition network
- The main algorithm is implemented in [baselines/ppo2/model.py](baselines/ppo2/model.py)
