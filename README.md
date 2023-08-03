

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

Github: https://github.com/Alexander-Nasuta/graph-jsp-env

PyPi: https://pypi.org/project/graph-jsp-env/

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

### Install the Package 
Install the package with pip:
```
   pip install graph-jsp-env
```
### Minimal Working Example
the code below shows a minimal working example without any reinforcement learning 
```
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
import numpy as np

jsp = np.array([
    [
        [0, 1, 2, 3],  # job 0
        [0, 2, 1, 3]  # job 1
    ],
    [
        [11, 3, 3, 12],  # task durations of job 0
        [5, 16, 7, 4]  # task durations of job 1
    ]

])
env = DisjunctiveGraphJspEnv(jps_instance=jsp)

# loop over all actions
for i in range(env.total_tasks_without_dummies):
    _ = env.step(i)
    env.render()
# schedule is done when every action/node is scheduled
env.render(wait=None)  # with wait=None the window remains open till a button is pressed
```

### Visualisations
The environment offers multiple visualisation options, some of which are shown below

![](https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/ganttAndGraph.png)
![](https://github.com/Alexander-Nasuta/graph-jsp-env/raw/master/resources/readme_images/console.png)

# Getting Started

If you just want to use the environment, then only the Usage section is relevant for you.
If you want to further develop the environment the follow the instructions in the Development section.

## Usage

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


