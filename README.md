## Model-Based Generative Adversarial Imitation Learning

Code for Paper "Model-Based Adversarial Inverse Reinforcement Learning", by Lalit Lal, Saad Saleem, Anson Leung.

## Dependencies (tested on macOS only, Python 3.6.10)
* Gym >= 0.18.0
* Tensorflow >= 1.15.0
* PyBullet-Gym (https://github.com/benelot/pybullet-gym)
* Garage (open-source RL Library) == 2020.6.3

## Running MAIRL
* In a new virtual env, first run
* ```pip install -r requirements.txt```
Run the following command to train the Pybullet-Gym Ant environment by imitating an expert trained with TRPO

```python
python main.py
```

## Generating TRPO Experts using Garage
* In a new virtual env, first run
* ``` pip install -r requirements_scripts.txt```
* You can train an expert using the following (be sure to specify environment in the script): ```python scripts/train_expert_policy.py```
* You can use the expert by referencing its name from the "data/local/experiment" directory in the following file and calling it: ```python scripts/run_expert_policy.py```
* Finally, you can generate expert data from an expert loading a model into and calling this script (similar to step above): ```python scripts/generate_minigrid_data.py```
