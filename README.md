# torch2tensorrt

### This repository is a simple way to convert a PyTorch model to Tensorrt, [increasing its performance up to 6x](https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/).

### Important: this repository only creates an environment for converting models. Code may need to be modified according to one's specific model
### Getting started
- In order to run a notebook with benchmarks run ``` docker compose --profile prototyping up ``` and run the notebook cells
- Run ``` docker compose --profile prod up ``` if you only want to save the models, without benchmarking
