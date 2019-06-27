# Nelder--Mead Method with Predictive Parallel Evaluation

## Usage

Install dependencies.

```
$ pip install -r requirements.txt
```

Run the example script.

```
$ python3 example.py --num_dim=2 --num_parallels 10 --num_montecarlo 100 --num_speculative_iter 3 --max_gp_samples 100
```

## Paper

```bibtex
@INPROCEEDINGS{ozaki-neldermead19,
author    = {Y. Ozaki and S. Watanabe and M. Onishi},
title     = {Accelerating the Nelder--Mead Method with Predictive Parallel Evaluation},
booktitle = {6th ICML Workshop on Automated Machine Learning},
year      = {2019},
month     = jun
}
```
