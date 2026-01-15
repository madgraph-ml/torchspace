<p align="center">
  <img src="docs/_static/logo-light.png" width="300", alt="TorchSpace">
</p>

<h3 align="center">Differentiable and GPU ready phase-space mappings</h3>

<p align="center">
<a href="https://arxiv.org/abs/2408.01486"><img alt="Arxiv" src="https://img.shields.io/badge/arXiv-2408.01486-b31b1b.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://pytorch.org"><img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.0-DC583A.svg?style=flat&logo=pytorch"></a>
</p>

This repository contains a refactored version of the code used in our publication *Differentiable MadNIS-Lite*. 

## Supported mappings

**TorchSpace** contains and supports a selection of different phase-space mappings which are useful for collider physics:
- Chili [https://arxiv.org/abs/2302.10449]
- Rambo [Comput. Phys. Commun. 40 (1986) 359-373]
- Rambo on diet [https://arxiv.org/abs/1308.2922]
- Mahambo [https://arxiv.org/abs/2305.07696]
- Diagram based [see this paper]
- 1 $\to$ 3  block [https://freidok.uni-freiburg.de/data/154629]
- including trainable variants of the above


## Installation

```sh
# clone the repository
git clone https://github.com/madgraph-ml/torchspace.git
# then simply install (editable if needed with flag "-e")
cd torchspace
pip install .
```

## Usage example

For an example usage of, for instance Mahambo, see `examples/rambo/rambo_example.py`

## Citation

If you use this code or parts of it, please cite:

    @article{Heimel:2024wph,
    author = "Heimel, Theo and Mattelaer, Olivier and Plehn, Tilman and Winterhalder, Ramon",
    title = "{Differentiable MadNIS-Lite}",
    eprint = "2408.01486",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "IRMP-CP3-24-23",
    doi = "10.21468/SciPostPhys.18.1.017",
    journal = "SciPost Phys.",
    volume = "18",
    number = "1",
    pages = "017",
    year = "2025"}

