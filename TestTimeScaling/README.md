# Test-time scaling method with CyclicReflex

## How to navigate this project 🧭

This project is simple by design and mostly consists of:

* [`scripts`](./scripts/) to scale test-time compute for open models. 
* [`recipes`](./recipes/) to apply different search algorithms at test-time. Three algorithms are currently supported: Best-of-N, beam search, and Diverse Verifier Tree Search (DVTS). Each recipe takes the form of a YAML file which contains all the parameters associated with a single inference run. 


## Getting Started

1. To run the code in this project, first, create a Python virtual environment using e.g. Conda:

      ```shell
      conda create -n sal python=3.11 && conda activate sal

      pip install -e '.[dev]'
      ```

2. Next, log into your Hugging Face account as follows:

      ```shell
      huggingface-cli login
      ```

3. Finally, install Git LFS so that you can push models to the Hugging Face Hub:

      ```shell
      sudo apt-get install git-lfs
      ```

4. You can now check out the `scripts` and `recipes` directories for instructions on how to scale test-time compute for open models!

## Project structure

```
├── LICENSE
├── Makefile                    <- Makefile with commands like `make style`
├── README.md                   <- The top-level README for developers using this project
├── recipes                     <- Recipe configs, accelerate configs, slurm scripts
├── scripts                     <- Scripts to scale test-time compute for models
├── pyproject.toml              <- Installation config (mostly used for configuring code quality & tests)
├── setup.py                    <- Makes project pip installable (pip install -e .) so `sal` can be imported
├── src                         <- Source code for use in this project
└── tests                       <- Unit tests
```


## Citation

If you find the content of this repo useful in your work, please cite it as follows via `\usepackage{biblatex}`:

```
@misc{beeching2024scalingtesttimecompute,
      title={Scaling test-time compute with open models},
      author={Edward Beeching and Lewis Tunstall and Sasha Rush},
      url={https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute},
}
```

Please also cite the original work by DeepMind upon which this repo is based:

```
@misc{snell2024scalingllmtesttimecompute,
      title={Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters}, 
      author={Charlie Snell and Jaehoon Lee and Kelvin Xu and Aviral Kumar},
      year={2024},
      eprint={2408.03314},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.03314}, 
}
```

