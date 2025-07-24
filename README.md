# FairTreeExperiments

This repository contains the codebase for the FairTree experiments. To ensure the project works seamlessly, you need to set the `PYTHONPATH` environment variable properly.

## Critical: Set up constants.py

This repository uses `constants.py` for local configuration. This file is ignored by Git, so changes remain local to your system.

To set it up:

1. Copy the template file:
   ```bash
   cp constants.py.template constants.py
   ```

2. Edit ```constants.py``` and replace placeholder values with your local path information:
   ```bash
   LOCAL_PATH = "/your/local/path"
   ```

## Setting Up PYTHONPATH

The `PYTHONPATH` environment variable must point to the root directory of this repository and the `FairTree` subdirectory. This allows Python to locate the necessary modules when running scripts or importing them.

### Steps to Set PYTHONPATH

After cloning the repository, follow these steps:

1. **Determine the Full Path to the Repository**  
   Replace `PATH_TO_REPO` in the instructions below with the absolute path to where you cloned this repository.

   Example: If you cloned the repository to `/home/user/projects/FairTreeExperiments`, use `/home/user/projects/FairTreeExperiments` as the `PATH_TO_REPO`.

2. **Set PYTHONPATH**

#### For Bash Users

To temporarily set `PYTHONPATH` for the current session:

```bash
export PYTHONPATH=PATH_TO_REPO:PATH_TO_REPO/FairTree
```

To make it permanent (persisting across sessions), add the following line to your ~/.bashrc (or ~/.bash_profile on macOS):

```bash
export PYTHONPATH=PATH_TO_REPO:PATH_TO_REPO/FairTree
```

After adding the line, reload your shell configuration:

```bash
source ~/.bashrc
```

#### For Fish Users

```fish
set -x PYTHONPATH PATH_TO_REPO PATH_TO_REPO/FairTree
```

To make it permanent (persisting across sessions), edit your Fish configuration file (~/.config/fish/config.fish) and add the following line:

```fish
set -x PYTHONPATH PATH_TO_REPO PATH_TO_REPO/FairTree
```

After editing, reload the configuration:

```fish
source ~/.config/fish/config.fish
```

### Verify PYTHONPATH

To ensure PYTHONPATH is set correctly, run the following command in your shell:

```bash
echo $PYTHONPATH
```

You should see output similar to:

```bash
/home/user/projects/FairTreeExperiments:/home/user/projects/FairTreeExperiments/FairTree
```

## Running the methods, hyperparamter optimization and evaluation

For running a single run of the one-tree or dual-tree approach, you have to use `algos_one_tree/method.py` or `algos_two_trees/method.py`.

The hyperparameter search can be performed using the script `run_hp_opt.sh` in the corresponding directory.

In the directory `evaluation`, you can find the python files that retrain the models for the best found hyperparameters, i.e., `one_tree_opt.py`, `two_tree_opt.py`, and `rel_threshold_optimizer_opt.py`.

Be careful to pass the correct paths as arguments.
