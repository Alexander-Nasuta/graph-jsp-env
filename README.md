

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
An [Gymnasium Environment](https://gymnasium.farama.org/) implementation 
of the Job Shop Scheduling Problem (JSP) using the disjunctive graph approach.

- **Github**: https://github.com/Alexander-Nasuta/graph-jsp-env

- **PyPi**: https://pypi.org/project/graph-jsp-env/

This environment is inspired by the 
[The disjunctive graph machine representation of the job shop scheduling problem](https://www.sciencedirect.com/science/article/pii/S0377221799004865)
by Jacek Błażewicz and
[Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf)
by Zhang, Cong, et al.

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
import stable_baselines3 as sb3
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
env = sb3.common.monitor.Monitor(env)


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


env = ActionMasker(env, mask_fn)

model = sb3_contrib.MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)

# Train the agent
log.info("training the model")
model.learn(total_timesteps=10_000)
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

## Development 
The following sections are only relevant if you plan on further develop the environment and introduce code changes into 
the environment itself.

To run this Project locally on your machine follow the following steps:

1. Clone the repo
   ```sh
   git clone https://github.com/Alexander-Nasuta/graph-jsp-env.git
   ```
2. Install the python requirements_dev packages. `requirements_dev.txt` includes all the packages of
specified `requirements.txt` and some additional development packages like `mypy`, `pytext`, `tox` etc. 
    ```sh
   pip install -r requirements_dev.txt
   ```
3. Install the modules of the project locally. For more info have a look at 
[James Murphy's testing guide](https://www.youtube.com/watch?v=DhUpxWjOhME)
   ```sh
   pip install -e .
   ```

### Testing

For testing make sure that the dev dependencies are installed (`requirements_dev.txt`) and the models of this 
project are set up (i.e. you have run `pip install -e .`).  

Then you should be able to run

```sh
mypy src
```

```sh
flake8 src
```

```sh
pytest
```

or everthing at once using `tox`.

```sh
tox
```

### IDEA

I recommend to use [Pycharm](https://www.jetbrains.com/de-de/pycharm/).
Of course any code editor can be used instead (like [VS code](https://code.visualstudio.com/) 
or [Vim](https://github.com/vim/vim)).

This section goes over a few recommended step for setting up the Project properly inside [Pycharm](https://www.jetbrains.com/de-de/pycharm/).

#### PyCharm Setup
1. Mark the `src` directory as `Source Root`.
```
   right click on the 'src' -> 'Mark directory as' -> `Source Root`
```

2. Mark the `resources` directory as `Resource Root`.
```
   right click on the 'resources' -> 'Mark directory as' -> `Resource Root`
```

3. Mark the `tests` directory as `Test Source Root`.
```
   right click on the 'tests' -> 'Mark directory as' -> `Test Source Root`
```

afterwards your project folder should be colored in the following way:

<div align="center">
  <a>
    <img src="https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/mark_project_folders.png"  height="320">
  </a>
</div>

4. (optional) When running a script enable `Emulate terminal in output console`
```
Run (drop down) | Edit Configurations... | Configuration | ☑️ Emulate terminal in output console
```

![](https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/colored_logs_settings.png)


# License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- MARKDOWN LINKS & IMAGES todo: add Github, Linked in etc.-->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[screenshot]: resources/readme_images/screenshot.png


