# ML Playground
A versatile Command Line Interface (CLI) tool designed for training different machine learning models on various environments and watching their progress.

## Features:
Kick off training for various models using a simple command.
Watch the progress of ongoing training sessions.
Integrated with wandb for powerful monitoring capabilities.

## Installation:
### 1. Clone the Repository:
```bash
git clone git@github.com:CraigSalajan/ml-playground.git
cd ml-playground
```

### 2. Set up a Virtual Environment (Optional but recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### 3. Install the Dependencies:
```bash
pip install -r requirements.txt
```

## Usage:
### 1. Training:
To initiate training, use the following command:

```bash
python -m training train -t <Trainer>
```

Optional Parameters:

-w: Enable wandb integration.

**Note:** This requires an environment variable WANDB_API_KEY with your wandb api key
```bash
python -m training train -t <Trainer> -w
```

-ts: Set the number of timesteps. Replace <num_timesteps> with your desired number.
```bash
python -m training train -t <Trainer> -ts <num_timesteps>
```

### 2. Watching Training Progress:
To watch the progress of a training session:

```bash
python -m training watch -r <run_id>
```
Replace <run_id> with the provided ID of the training run you wish to observe.

## Contributing:
We welcome contributions! Please see the CONTRIBUTING.md file for guidelines

License:
This project is licensed under MIT. See the LICENSE file for details.