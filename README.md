## Sparse Neural Network implementations
```python
import paddle.sparse

structure  = paddle.sparse.CachedLayeredGraph()
# .. add nodes & edges to the networkx graph structure

# Build a neural network classifier with 784 input and 10 output neurons and the given structure
model = paddle.sparse.MaskedDeepDAN(784, 10, structure)
model.apply_mask()  # Apply the mask on the weights (hard, not undoable)
model.recompute_mask()  # Use weight magnitude to recompute the mask from the network
model.get_structure()  # Get the structure -- a networkx graph -- based on the current mask
```
```python
import paddle.sparse

model = paddle.sparse.MaskedDeepFFN(784, 10, [100, 100])
# .. train model
model.get_structure()  # a networkx graph
``` 


# Paddle - Passau Data Science Deep Learning Environments
**Paddle** is in progress and will merge acquired knowledge from PADDLL library, kayak and several jupyter notebooks into one standard library for wrapping tensorflow/pytorch implementations for Sparse Neural Network and Neuroevolution computations.

# Usage
## Install from private repository
```bash
pip install --upgrade git+ssh://git@gitlab.padim.fim.uni-passau.de:13003/paddle/paddle.git
```


## Training
..


# Development

## Architecture

## Project Structure
- following [Hitchhikers Guide to Python](http://docs.python-guide.org/en/latest/writing/structure/)
