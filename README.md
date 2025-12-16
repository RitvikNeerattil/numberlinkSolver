# NumberLink Puzzle Solver using DeepXube

This project integrates the `NumberLink` puzzle environment with `DeepXube`, a framework for solving pathfinding problems using deep reinforcement learning and heuristic search.

The integration allows `DeepXube` to be used to train agents that can solve `NumberLink` puzzles.

## Usage

Here is a simple example of how to instantiate the `NumberLink` environment within the `DeepXube` framework:

```python
from deepxube.implementations.numberlink import NumberLinkDeepXubeEnv

# Get environment
env = NumberLinkDeepXubeEnv(width=7, height=7, num_colors=5)

# Now you can use the env with DeepXube's training and search algorithms
```
