# Examples: MLP Training and Analysis

This folder contains examples for training and analyzing MLPs on synthetic data from the 10-label, 5-dimension distribution.

## Files

- `is_demo_10_labels_5_dims.py` - Main synthetic data generation with 10 labels and 5 dimensions
- `train_mlp_comparison.py` - Train MLP and compare with analytical loss bounds
- `plot_results.py` - Visualize training results and compare with theoretical bounds

## Usage

### Training an MLP

Train an MLP with default settings (15 epochs, 50K samples, all 10 labels):

```bash
cd /path/to/benchmark-model
PYTHONPATH=. python bm/examples/train_mlp_comparison.py --save_results results_directory
```

Train with specific parameters:

```bash
PYTHONPATH=. python bm/examples/train_mlp_comparison.py \
    --epochs 20 \
    --num_samples 100000 \
    --labels "0,1,2,3,4" \
    --save_results results_5_labels
```

### Plotting Results

Plot training results and analytical comparison:

```bash
PYTHONPATH=. python bm/examples/plot_results.py results_directory/mlp_results.json
```

Compare multiple experiments:

```bash
PYTHONPATH=. python bm/examples/plot_results.py results_1/mlp_results.json \
    --compare results_2/mlp_results.json results_3/mlp_results.json
```

### Generating Synthetic Data Only

Generate synthetic data without training:

```bash
cd bm/examples
PYTHONPATH=../.. python is_demo_10_labels_5_dims.py \
    --num_samples 10000 \
    --labels "0,1,2" \
    --save_dir output_directory
```

## Parameters

### train_mlp_comparison.py

- `--epochs` (int): Number of training epochs (default: 15)
- `--num_samples` (int): Number of data samples (default: 50,000)
- `--labels` (str): Comma-separated label indices 0-9 (default: all labels)
- `--save_results` (str): Directory to save results (optional)
- `--seed` (int): Random seed (default: 1234)

### plot_results.py

- `results_file` (str): Path to JSON results file from training
- `--compare` (list): Additional result files for comparison plotting

## MLP Architecture

The MLP has the following architecture:
- Input: 5 dimensions (continuous features)
- Hidden layers: 3 layers of 1024 units each with ReLU activation
- Output: 10 classes (or subset as specified by `--labels`)
- Total parameters: ~2.1M

## Training Details

- Optimizer: SGD with momentum (lr=0.01, momentum=0.9)
- Loss: Cross-entropy loss
- Batch size: 256
- Train/validation split: 80/20

## Key Metrics

The scripts compute and compare:

1. **Analytical Loss**: Theoretical optimal loss based on mutual information
2. **MLP Loss**: Achieved cross-entropy loss during training
3. **Loss Ratio**: MLP loss divided by analytical loss (closer to 1.0 is better)
4. **Theoretical Accuracy**: Estimated maximum achievable accuracy
5. **MLP Accuracy**: Actual validation accuracy achieved

## Unit Conversion

- Analytical loss is computed in **bits**
- PyTorch loss is computed in **nats**
- Conversion: `nats = bits × ln(2) ≈ bits × 0.693`

## Example Output

```
============================================================
RESULTS COMPARISON
============================================================
Analytical loss:    0.4817 bits = 0.3339 nats
MLP final loss:     0.7978 bits = 0.5530 nats
Loss difference:    0.2191 nats
Loss ratio (MLP/Analytical): 1.656

Theoretical max accuracy: 0.8267
MLP final accuracy:       0.8468
Accuracy difference:      0.0201

✅ MLP reached near-optimal accuracy (within 10% of theoretical)
❌ MLP did not reach analytical loss
```

## Generated Files

Training saves:
- `mlp_results.json` - Complete training history and metrics
- `mlp_training_analysis.png` - Training progress plots (when plotting)

Data generation saves:
- `data.npz` - X and Y samples as NumPy arrays
- `metadata.json` - Distribution metadata and mutual information values 