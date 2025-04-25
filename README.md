[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15044111.svg)](https://doi.org/10.5281/zenodo.15044111)
![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=flat)
[![PyPI version](https://img.shields.io/pypi/v/graph-jsp-env)](https://pypi.org/project/graph-jsp-env/)
![License](https://img.shields.io/pypi/l/graph-jsp-env)
[![Documentation Status](https://readthedocs.org/projects/graph-jsp-env/badge/?version=latest)](https://graph-jsp-env.readthedocs.io/en/latest/?badge=latest)


<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://cybernetics-lab.de/">
    <img src="https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/logo.png" alt="Logo" height="80">
  </a>

  <h1 align="center">
     Graph Job Shop Problem Gym Environment 
  </h1>

   <a>
    <img src="https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/graph_jsp_tikz.png" alt="Logo" height="180">
  </a>

</div>




# About The Project
A [Gymnasium Environment](https://gymnasium.farama.org/) implementation 
of the Job Shop Scheduling Problem (JSP) using the disjunctive graph approach.

- **Github**: https://github.com/Alexander-Nasuta/graph-jsp-env
- **PyPi**: https://pypi.org/project/graph-jsp-env/
- **Documentation**: https://graph-jsp-env.readthedocs.io/en/latest/

This environment is inspired by the 
[The disjunctive graph machine representation of the job shop scheduling problem](https://www.sciencedirect.com/science/article/pii/S0377221799004865)
by Jacek Błażewicz and
[Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf)
by Zhang et al.

This environment does not explicitly include disjunctive edges, like specified by Jacek Błażewicz, 
only conjunctive edges. 
Additional information is saved in the edges and nodes, such that one could construct the disjunctive edges, so the is no loss in information.

This environment is more similar to the Zhang, Cong, et al. implementation.
Zhang, Cong, et al. seems to store exclusively time-information exclusively inside nodes 
(see Figure 2: Example of state transition) and no additional information inside the edges (like weights in the representation of Jacek Błażewicz).

The DisjunctiveGraphJssEnv uses the `networkx` library for graph structure and graph visualization.
It is highly configurable and offers various rendering options.

# Quick Start

Install the package with pip:
```
   pip install graph-jsp-env
```

## Minimal Working Example: Random Actions
The code below shows a minimal working example without any reinforcement learning 
```python
import numpy as np
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

jsp = np.array([
    [[1, 2, 0],  # job 0
     [0, 2, 1]],  # job 1
    [[17, 12, 19],  # task durations of job 0
     [8, 6, 2]]  # task durations of job 1
])

env = DisjunctiveGraphJspEnv(
    jps_instance=jsp,
    perform_left_shift_if_possible=True, 
    normalize_observation_space=True,  # see documentation of DisjunctiveGraphJspEnv::get_state for more information
    flat_observation_space=True,  # see documentation of DisjunctiveGraphJspEnv::get_state for more information
    action_mode='task',  # alternative 'job'
    dtype='float32'  # dtype of the observation space
)

terminated = False
info = {}
for i in range(6):
    # get valid action mask. sample expects it to be a numpy array of type int8
    mask = np.array(env.valid_action_mask()).astype(np.int8)
    action = env.action_space.sample(mask=mask)
    state, reward, terminated, truncated, info = env.step(action)
    # chose the visualisation you want to see using the show parameter
    # console rendering
    env.render(show=["gantt_console", "graph_console"])
    
print(f"makespan: {info['makespan']}")
```
# Stable Baselines3
To run the example below you need to install the following packages:

```pip install stable_baselines3```

```pip install sb3_contrib```

It is recommended to use the `MaskablePPO` algorithm from the `sb3_contrib` package.

```python
import gymnasium as gym
import sb3_contrib
import numpy as np
from stable_baselines3.common.monitor import Monitor

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from graph_jsp_env.disjunctive_graph_logger import log
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

jsp = np.array([
   [[1, 2, 0],  # job 0
    [0, 2, 1]],  # job 1
   [[17, 12, 19],  # task durations of job 0
    [8, 6, 2]]  # task durations of job 1
])

env = DisjunctiveGraphJspEnv(
   jps_instance=jsp,
   perform_left_shift_if_possible=True,
   normalize_observation_space=True,
   flat_observation_space=True,
   action_mode='task',  # alternative 'job'
)
env = Monitor(env)


def mask_fn(env: gym.Env) -> np.ndarray:
   return env.unwrapped.valid_action_mask()


env = ActionMasker(env, mask_fn)

model = sb3_contrib.MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)

# Train the agent
log.info("training the model")
model.learn(total_timesteps=10_000)
```

# Ray rllib

The following example was provided by [@nhuet](https://github.com/nhuet). 
To run the example below you need to install the following packages:

```pip install "ray[rllib]" torch "gymnasium[atari,accept-rom-license,mujoco]"```

```python
import numpy as np
import ray
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from ray.rllib.algorithms import PPO
from ray.tune import register_env

jsp = np.array(
    [
        [
            [0, 1, 2],  # machines for job 0
            [0, 2, 1],  # machines for job 1
            [0, 1, 2],  # machines for job 2
        ],
        [
            [3, 2, 2],  # task durations of job 0
            [2, 1, 4],  # task durations of job 1
            [0, 4, 3],  # task durations of job 2
        ],
    ]
)

register_env(
    "jsp",
    lambda env_config: DisjunctiveGraphJspEnv(
        jps_instance=jsp,
        visualizer_kwargs=dict(handle_stop_signals=False)
    ),
)

ray.init()
algo = PPO(config=PPO.get_default_config().environment("jsp"))
algo.train()
```



### Visualisations
The environment offers multiple visualisation options.
There are four visualisations that can be mixed and matched:
- `gantt_window`: a gantt chart visualisation in a separate window
- `graph_window`: a graph visualisation in a separate window. This visualisation is computationally expensive.
- `gantt_console`: a gantt chart visualisation in the console
- `graph_console`: a graph visualisation in the console

The desired visualisation can be defaulted in the constructor of the environment with the argument `default_visualisations`.
To enable all visualisation specify `default_visualisations=["gantt_window", "gantt_console", "graph_window", "graph_console"]`.
The default visualisations are the used by the `render()` method if no visualisations are specified (using the `show` argument).

## Visualisation in OpenCV Window
This visualisation can enabled by setting `render_mode='window'` or setting the argument `default_visualisations=["gantt_window", "graph_window"]` in the constructor of the environment.
Additional parameters for OpencCV will be passed to the `cv2.imshow()` function.
Example:
```python
env.render(wait=1_000)  # render window closes automatically after 1 seconds
env.render(wait=None) # render window closes when any button is pressed (when the render window is focused)
```

![](https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/ft06_window_presi.gif)

## Console Visualisation 
This visualisation can enabled by setting `render_mode='window'` or setting the argument `default_visualisations=["gantt_console", "graph_console"]` in the constructor of the environment.
![](https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/ft06_console.gif)


## More Examples

Various examples can be found in the [graph-jsp-examples](https://github.com/Alexander-Nasuta/graph-jsp-examples) repo.

## State of the Project

This project is complementary material for a research paper.
It will not be frequently updated.
Minor updates might occur.

## Dependencies

This project specifies multiple requirements files.
`requirements.txt` contains the dependencies for the environment to work. These requirements will be installed automatically when installing the environment via `pip`.
`requirements_dev.txt` contains the dependencies for development purposes. It includes the dependencies for testing, linting, and building the project on top of the dependencies in `requirements.txt`.

In this Project the dependencies are specified in the `pyproject.toml` file with as little version constraints as possible.
The tool `pip-compile` translates the `pyproject.toml` file into a `requirements.txt` file with pinned versions.
That way version conflicts can be avoided (as much as possible) and the project can be built in a reproducible way.

## Development Setup

If you want to check out the code and implement new features or fix bugs, you can set up the project as follows:

### Clone the Repository

clone the repository in your favorite code editor (for example PyCharm, VSCode, Neovim, etc.)

using https:
```shell
git clone https://github.com/Alexander-Nasuta/graph-jsp-env
```
or by using the GitHub CLI:
```shell
gh repo clone Alexander-Nasuta/graph-jsp-env
```

if you are using PyCharm, I recommend doing the following additional steps:

- mark the `src` folder as source root (by right-clicking on the folder and selecting `Mark Directory as` -> `Sources Root`)
- mark the `tests` folder as test root (by right-clicking on the folder and selecting `Mark Directory as` -> `Test Sources Root`)
- mark the `resources` folder as resources root (by right-clicking on the folder and selecting `Mark Directory as` -> `Resources Root`)

at the end your project structure should look like this:

todo

### Create a Virtual Environment (optional)

Most Developers use a virtual environment to manage the dependencies of their projects.
I personally use `conda` for this purpose.

When using `conda`, you can create a new environment with the name 'my-graph-jsp-env' following command:

```shell
conda create -n my-graph-jsp-env python=3.11
```

Feel free to use any other name for the environment or an more recent version of python.
Activate the environment with the following command:

```shell
conda activate my-graph-jsp-env
```

Replace `my-graph-jsp-env` with the name of your environment, if you used a different name.

You can also use `venv` or `virtualenv` to create a virtual environment. In that case please refer to the respective documentation.

### Install the Dependencies

To install the dependencies for development purposes, run the following command:

```shell
pip install -r requirements_dev.txt
pip install tox
```

The testing package `tox` is not included in the `requirements_dev.txt` file, because it sometimes causes issues when
using github actions.
Github Actions uses an own tox environment (namely 'tox-gh-actions'), which can cause conflicts with the tox environment on your local machine.

Reference: [Automated Testing in Python with pytest, tox, and GitHub Actions](https://www.youtube.com/watch?v=DhUpxWjOhME).

### Install the Project in Editable Mode

To install the project in editable mode, run the following command:

```shell
pip install -e .
```

This will install the project in editable mode, so you can make changes to the code and test them immediately.

### Run the Tests

This project uses `pytest` for testing. To run the tests, run the following command:

```shell
pytest
```

For testing with `tox` run the following command:

```shell
tox
```

Tox will run the tests in a separate environment and will also check if the requirements are installed correctly.

### Building and Publishing the Project to PyPi

In order to publish the project to PyPi, the project needs to be built and then uploaded to PyPi.

To build the project, run the following command:

```shell
python -m build
```

It is considered good practice use the tool `twine` for checking the build and uploading the project to PyPi.
By default the build command creates a `dist` folder with the built project files.
To check all the files in the `dist` folder, run the following command:

```shell
twine check dist/**
```

If the check is successful, you can upload the project to PyPi with the following command:

```shell
twine upload dist/**
```

### Documentation
This project uses `sphinx` for generating the documentation.
It also uses a lot of sphinx extensions to make the documentation more readable and interactive.
For example the extension `myst-parser` is used to enable markdown support in the documentation (instead of the usual .rst-files).
It also uses the `sphinx-autobuild` extension to automatically rebuild the documentation when changes are made.
By running the following command, the documentation will be automatically built and served, when changes are made (make sure to run this command in the root directory of the project):

```shell
sphinx-autobuild ./docs/source/ ./docs/build/html/
```

This project features most of the extensions featured in this Tutorial: [Document Your Scientific Project With Markdown, Sphinx, and Read the Docs | PyData Global 2021](https://www.youtube.com/watch?v=qRSb299awB0).



## Contact

If you have any questions or feedback, feel free to contact me via [email](mailto:alexander.nasuta@wzl-iqs.rwth-aachen.de) or open an issue on repository.


