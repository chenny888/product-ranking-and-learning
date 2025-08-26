# Revenue Management with Ranking Learning: Replication Package

This repository contains the code and data for reproducing the numerical experiments from the paper "Revenue Management with Ranking Learning" submitted to Operations Research.

## Overview

The code implements algorithms for revenue management problems where customers have limited attention spans and make sequential purchasing decisions. The repository includes:

- **Section 6.1**: Offline experiments comparing different ranking algorithms
- **Section 6.2**: Online learning experiments with unknown attention parameters
- **Utility functions**: Core algorithms for assortment optimization and ranking

## Repository Structure

```
├── README.md                           # This file
├── pyproject.toml                      # Dependency management with uv
├── Offline_Experiments_Section_6_1.ipynb  # Jupyter notebook for offline experiments
├── Online_experiment_Section_6_2.py       # Python script for online experiments
├── utils.py                               # Core utility functions and algorithms
├── online_data_iter_*.pickle              # Pre-generated data from online experiments
└── ranking_OR_review_round_2.pdf          # Research paper
```

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Quick Start

### Option 1: Using uv (Recommended)

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   ```

2. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd revenue-management-experiments
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

### Option 2: Using pip

If you prefer using pip, install the dependencies from the pyproject.toml:

```bash
pip install numpy matplotlib scikit-learn pandas scipy jupyter
```

## Reproducing the Experiments

### Section 6.1: Offline Experiments

The offline experiments compare the performance of different ranking algorithms under various scenarios.

1. **Start Jupyter notebook**:
   ```bash
   # If using uv
   uv run jupyter notebook
   
   # If using pip
   jupyter notebook
   ```

2. **Open and run** `Offline_Experiments_Section_6_1.ipynb`

3. **Execute all cells** to reproduce the offline experimental results

The notebook will:
- Generate synthetic product data with different revenue and choice parameters
- Compare various ranking algorithms (Optimal, Approximate, Greedy, etc.)
- Produce performance comparison plots and tables

### Section 6.2: Online Learning Experiments

The online experiments simulate learning customer attention parameters in real-time.

1. **Run the online experiment script**:
   ```bash
   # If using uv
   uv run python Online_experiment_Section_6_2.py
   
   # If using pip
   python Online_experiment_Section_6_2.py
   ```

2. **Monitor progress**: The script runs 10 independent simulations (iterations), each with 10,000 customers

3. **Output**: The script generates pickle files `online_data_iter_*.pickle` containing:
   - Estimation errors for attention parameters
   - Cumulative revenues over time
   - Comparison with known-parameter benchmarks

**Note**: The complete online experiments may take several hours to run. Pre-computed results are included as `online_data_iter_*.pickle` files.

## Key Algorithms Implemented

### Core Functions (utils.py)

- **`AssortOpt(r, p, m)`**: Dynamic programming solution for assortment optimization
- **`ApproxOpt(r, p, G)`**: Approximation algorithm for ranking with attention constraints  
- **`TrueReward(S, r, p, G)`**: Calculates expected revenue for a given ranking
- **`UpperBound(R, G)`**: Computes theoretical upper bounds on performance

### Parameters

- **`r`**: Revenue vector for each product
- **`p`**: Choice probability vector for each product
- **`G`**: Attention decay vector (decreasing sequence representing customer attention)
- **`M`**: Attention span length
- **`N`**: Number of products

## Expected Results

### Offline Experiments (Section 6.1)
- Performance comparison plots showing approximation ratios
- Tables with numerical results for different problem instances
- Analysis of algorithm performance under various parameter settings

### Online Experiments (Section 6.2) 
- Learning curves showing parameter estimation accuracy over time
- Revenue performance comparison between learning and benchmark algorithms
- Convergence analysis of the online learning algorithm

## Computational Requirements

- **Memory**: ~2-4 GB RAM recommended
- **Runtime**: 
  - Offline experiments: ~10-30 minutes
  - Online experiments: ~2-6 hours for all iterations
- **Storage**: ~50 MB for all generated data files

## Dependencies

The project uses the following main packages:

- `numpy`: Numerical computing
- `matplotlib`: Plotting and visualization  
- `scikit-learn`: Machine learning algorithms (Linear Regression)
- `pandas`: Data manipulation
- `scipy`: Scientific computing (statistics functions)
- `jupyter`: Interactive notebook environment

All dependencies are managed through the `pyproject.toml` file with version pinning for reproducibility.

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure the virtual environment is activated and all dependencies are installed
2. **Memory issues**: Reduce the number of products (N) or simulation length (T) in the scripts
3. **Slow performance**: The online experiments are computationally intensive; consider running on a machine with multiple cores

### Contact

For questions about the code or experiments, please refer to the paper or contact the authors.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{chen2025revenue,
  title={Revenue maximization and learning in product ranking},
  author={Chen, Ningyuan and Li, Anran and Yang, Shuoguang},
  journal={Operations Research},
  volume={Accepted},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.