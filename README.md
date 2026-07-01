# 🏠 Housing Price Prediction

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **California housing price prediction using machine learning with feature engineering.**

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Dependencies](#-dependencies)
- [License](#-license)

## 🎯 Overview

This project predicts median house values in California using various machine learning algorithms. It demonstrates a complete ML pipeline from data loading to model evaluation.

### Key Highlights

- 📊 **Dataset**: California Housing Dataset (20,640 samples, 8 features)
- 🤖 **Algorithms**: Multiple regression models compared
- 🔧 **Feature Engineering**: Data preprocessing and transformation
- 📈 **Evaluation**: RMSE, MAE, R² metrics

## ✨ Features

- **Data Processing**: Automated data cleaning and preprocessing
- **Multiple Models**: Comparison of various regression algorithms
- **Feature Engineering**: Automatic feature transformation
- **Cross-Validation**: K-fold cross-validation for robust evaluation
- **Visualization**: Results plotting and analysis

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Steps

```bash
# Clone the repository
git clone https://github.com/Arefmtl/Housing-Price-Prediction.git
cd Housing-Price-Prediction

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Quick Start

```bash
# Run the main prediction script
python Housing.py
```

### Jupyter Notebook

```bash
# Launch Jupyter Notebook
jupyter notebook Housing.ipynb
```

### Example Code

```python
import pandas as pd
from Tool_box import DataProcessingTool, RegressionTool, ModelEvaluationTool

# Load data
processor = DataProcessingTool()
data = processor.load_data("Dataset/housing.csv")

# Prepare data for ML
processed_data = processor.prepare_data_for_ml(
    data,
    target_column='median_house_value',
    test_size=0.2,
    preprocessing_steps=['clean', 'encode', 'scale']
)

# Train models
regressor = RegressionTool()
models = regressor.train_multiple_models(
    processed_data['X_train'],
    processed_data['y_train']
)

# Evaluate models
evaluator = ModelEvaluationTool()
results = evaluator.evaluate_regression_models(
    models,
    processed_data['X_test'],
    processed_data['y_test']
)
```

## 📁 Project Structure

```
Housing-Price-Prediction/
├── Housing.py          # Main prediction script
├── Housing.ipynb       # Jupyter notebook with analysis
├── Housing.md         # Detailed documentation
├── Dataset/
│   └── housing.csv    # California housing dataset
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## 📊 Results

The project compares multiple regression algorithms:

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | - | - | - |
| Random Forest | - | - | - |
| Gradient Boosting | - | - | - |
| XGBoost | - | - | - |

*Run the script to see actual results*

## 🛠️ Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical visualization (optional)

## 🔧 ML Tools

This project uses tools from [TOOL-BOX](https://github.com/Arefmtl/TOOL-BOX) — a comprehensive ML toolbox with 8 specialized modular tools.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

**Ali Kazemi** - [@Arefmtl](https://github.com/Arefmtl)

Project Link: [https://github.com/Arefmtl/Housing-Price-Prediction](https://github.com/Arefmtl/Housing-Price-Prediction)
