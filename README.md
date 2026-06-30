# Machine Learning Optimal Quantum Error Correction Thresholds

This repository contains code for the paper *"Machine Learning Optimal Quantum Error Correction Thresholds"*.

## Virtual Environment Setup

To activate the virtual environment, run:

```bash
source .venv/bin/activate
```

## Code Overview

The code is divided into three parts, one each for depolarizing noise in the code capacity setting, phenomenological noise, and circuit-level noise. We learn a neural network to estimate the coherent information by estimating the densities $$p(l|s)$$ for syndromes $$s$$ and logical operators $$l$$.

## Running the scripts
To execute the scripts from the terminal, run e.g. the following command that trains a transformer model for the distance-three surface code.
```
python3 main.py --distance 3 --noise 0.05 --noise_model "depolarizing" --task "train"
```

## Command-Line Arguments

Below are the available command-line arguments:

### Required Arguments:
- `--noise_model`  
  Select the noise model, which is simulated. Options:
  - `"bitflip"`
  - `"depolarizing"`
  - `"phenomenological"`
  - `"circuit-level"`

- `--task`  
  Define the task to be executed. Options:
  - `"train"`: Train the model
  - `"decoding"`: Perform decoding
  - `"plot_ci"`: Plot the coherent information estimates
  - `"plot_log_error"`: Plot logical error rate
  - `"attention"`: Visualize attention weights
  - `"loss"`: Plot loss function during training
  - `"plot-single-latent"`: Plot a single latent space representation
  - `"reconstruction"`: Visualize reconstructed data
  - `"raw-data"`: Visualize raw data
  - `"plot-latent-reconstruction"`: Plot latent space reconstruction
  - `"plot-vs-raw"`: Plot comparison between latent space and raw data

### Additional Options:
- `--distance`  
  The distance of the surface code (e.g., 3 for distance-three surface code).
  
- `--noise`  
  The noise rate (e.g., 0.05).

Available distances and noise rates can be read in the paper.

## Data and Output
The main scripts for the learning task can be found in main_qec.py. All model checkpoints and data is saved in data, and plots are saved in plots.


 
