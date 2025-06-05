# Deep Learning-Based Estimation of UAV-Swarm Communication Constraints

This project implements a deep learning approach to estimate key communication parameters in UAV swarms using high-level, easily measurable metrics. The system predicts communication failure rate and maximum communication range by leveraging formation stability indicators such as shape error, Raft leader absence, and presence consistency, combined with swarm size and estimated communication delay.

## Description

The project addresses the challenge of estimating communication parameters in UAV swarms through a data-driven approach. Key features include:

1. A custom Temporal Summary Layer that compresses extensive time series into multi-scale features
2. Deep learning models that capture both short-term fluctuations and long-term trends
3. Non-intrusive monitoring of swarm communication health
4. High predictive accuracy with R² scores exceeding 0.83

## Main Features

### Data Processing
- Time series compression using Temporal Summary Layer
- Feature importance analysis via SHAP
- Data normalization and preprocessing
- Custom windowing strategies (short, linear, long)

### Model Architecture
- Feedforward neural network with 4 hidden layers (256, 128, 64, 32 units)
- LeakyReLU activations with dropout regularization
- Smooth L1 (Huber) loss function
- AdamW optimizer with learning rate scheduling

### Analysis Tools
- Feature importance visualization
- Loss and R² Score analysis
- Temporal pattern analysis
- Model performance evaluation

## Requirements

```bash
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
tensorflow>=2.8.0
scikit-learn>=0.24.0
seaborn>=0.11.0
tqdm>=4.62.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Project-GrADyS/DeepLearningEstimation.git
cd deep-learning-estimation
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Generation and Preprocessing
```bash
python data_generation.py
python data_preprocessing.py
```

### 2. Model Training
```bash
python model_training.py
```

### 3. Utilities
```bash
# Feature importance analysis
python feature_importance.py

# GPU configuration and management
python util_disable_GPU.py

# Data inspection and validation
python util_inspection_data_collected.py

# Results visualization
python util_loss_R2_graph.py
```

## Project Structure

```
deep-learning-estimation/
│
├── data_generation.py              # Synthetic data generation
├── data_preprocessing.py           # Data preprocessing
├── model_training.py               # Base model training
├── feature_importance.py           # Feature importance analysis
├── util_loss_R2_graph.py          # Results visualization
├── util_inspection_data_collected.py # Data inspection
├── util_disable_GPU.py            # GPU configuration if necessary
├── parameters.py                   # Project parameters
├── protocol.py                     # Experiment protocol
│
├── dataset_training/              # Training data
├── dataset_validation/            # Validation data
├── dataset_test/                  # Test data
├── models/                        # Saved models
├── plots/                         # Generated plots
│
├── dataset.csv                    # Main dataset
├── dataset.npz                    # Dataset in numpy format
├── k_fold_tests.csv              # K-fold test results
├── Feature_Importance_Table.csv   # Feature importance table
│
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## Data Formats

### Main Dataset (dataset.csv)
- Contains raw time series data
- Includes features and targets
- Recorded at 5-ms intervals over 5 minutes

### K-fold Test Results (k_fold_tests.csv)
- Windows: Window size
- Mode: Model type (short, linear, long)
- Test Loss (mean): Mean test loss
- R2 Score (mean): Mean R² score

## 🔍 Results Analysis

The project includes several tools for results analysis:
1. Comparative Loss and R² Score graphs
2. Feature importance analysis
3. Model performance tables
4. Time series visualizations

### Performance Metrics
- Test Loss: 0.0043
- R² Score (comm_range): 0.8315
- R² Score (comm_failure): 0.8927

## Contributing

Contributions are welcome! To contribute:

1. Fork the project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Laércio Lucchesi - laercio.lucchesi@gmail.com

Project Link: [https://github.com/Project-GrADyS/DeepLearningEstimation](https://github.com/Project-GrADyS/DeepLearningEstimation)

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{your-paper-2024,
  title={Deep Learning-Based Estimation of UAV-Swarm Communication Constraints},
  author={Author Names},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
``` 