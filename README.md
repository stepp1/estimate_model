# Benchmarking Models with Synthetic Data with Information-Theoretic Properties

This repository contains the code for generating synthetic data with information-theoretic properties used in the paper "Benchmarking Models with Synthetic Data with Information-Theoretic Properties". (WIP xD)

## Installation

We use `uv` to install the dependencies.

```bash
uv sync
```

If you need to install `uv`, refer to the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Development

- Main Code: 
  - Located at `bm/enhanced_synthetic.py`
  - Contains the implementation of `EnhancedSyntheticDistribution` which lets us generate synthetic data from a real distribution.
  - Details about its functionality can be found in the technical report.

- Fitting distributions:
  - Located at `bm/distributions/*`
  - This folder contains the code for the different distributions used in the benchmark model.
  - Each distribution has its own folder and its own `fit.py` file.
  - To run the code for a specific distribution, you can use the following command:

    ```bash
    python -m bm.distributions.<distribution>.fit --<parameter_name>=<parameter_value>
    ```
  - Example to generate data and visualizations for the Synthetic MNIST distribution:
    ```bash
    python -m bm.distributions.<distribution>.fit -g -viz
    ```
    This saves them in the `data` and `figures` folders respectively in `bm/distributions/<distribution>/`.

- Already fitted distributions:
  - MNIST: located at `bm/distributions/mnist/`


## Fitting distributions via CLI:

To run the fitting script for a specific distribution, you can use the following command:

```bash
python -m bm.distributions.<distribution>.fit --<parameter_name>=<parameter_value>
```

Some common arguments are 

- `-g` or `--generate-data`: Generate data
- `-viz` or `--perform-visualization`: Generate visualizations
- `-dmi` or `--detailed-mi-computation`: Perform detailed mutual information computation
- `-fmi` or `--full-mi-computation`: Perform full mutual information computation
- `-rt` or `--report-time`: Report execution time
- `-md` or `--max-digit`: Maximum digit to consider (default: 9)
- `-ds` or `--data-size`: Size of the data (default: 28)
- `-ns` or `--num-samples`: Number of samples to generate (default: 100,000)
- `-nl` or `--num-levels`: Number of levels (default: 3)
- `-nb` or `--num-bases`: Number of bases (default: 2)
- `-bmax` or `--base-max-complexity`: Base maximum complexity (default: 3)
- `-npb` or `--num-prob-bases`: Number of probability bases (default: 2)
- `-cn` or `--corruption-name`: Name of corruption to apply (default: "identity")


