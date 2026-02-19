# Artemis - Machine Learning Library for Solidity

[![Solidity](https://img.shields.io/badge/Solidity-^0.8.0-blue.svg)](https://docs.soliditylang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Foundry](https://img.shields.io/badge/Built%20with-Foundry-orange.svg)](https://getfoundry.sh/)
[![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)]()
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

> **Comprehensive, modular, and well-documented machine learning library for educational purposes on the Ethereum blockchain**

Artemis adalah library machine learning komprehensif yang ditulis dalam Solidity, terinspirasi oleh scikit-learn dan framework ML modern. Library ini dirancang untuk demonstrasi dan pembelajaran tentang implementasi algoritma machine learning on-chain.

**⚠️ Important Note:** Library ini ditujukan untuk **educational purposes** dan **proof-of-concept** implementasi machine learning on-chain. Untuk penggunaan production, pertimbangkan carefully gas costs dan computational limitations.

## 📋 Table of Contents

- [Features Overview](#features-overview)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Gas Optimization](#gas-optimization)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Features Overview

Artemis menyediakan implementasi lengkap berbagai algoritma machine learning dengan arsitektur modular:

### 🔍 Supervised Learning Models

- **Linear Regression** - Regresi linear dengan gradient descent optimization
- **Logistic Regression** - Klasifikasi biner dengan sigmoid activation
- **K-Nearest Neighbors** - Instance-based learning untuk classification dan regression

### 🎯 Unsupervised Learning Models

- **K-Means Clustering** - Clustering dengan multiple initialization methods

### 🧠 Neural Network Framework

- **Neural Network** - Multi-layer perceptron dengan backpropagation
- **Dense Layers** - Fully connected layers dengan bias
- **Activation Layers** - Non-linear transformations
- **Optimizers** - SGD dan Adam optimizers

### 📊 Mathematical Utilities

- **Array Utils** - Operasi array dan vektor
- **Matrix Operations** - Operasi matriks dasar
- **Statistics** - Fungsi statistik dan analisis data
- **Data Preprocessing** - Normalisasi dan scaling

### ⚙️ Core Components

- **Activation Functions** - ReLU, Sigmoid, Softmax
- **Loss Functions** - MSE, MAE, Cross Entropy
- **Optimizers** - SGD, Adam dengan momentum
- **Interfaces** - Standardized interfaces untuk ekstensibilitas

## 📥 Installation Guide

### Prerequisites

- [Foundry](https://getfoundry.sh/) - Toolkit development Ethereum
- Solidity ^0.8.0

### Project Structure

```
Artemis/
├── src/
│   └── Artemis/
│       ├── activation/          # Activation functions (ReLU, Sigmoid, Softmax)
│       ├── examples/            # Complete usage examples
│       ├── interfaces/          # Core interfaces (IModel, IActivation, etc.)
│       ├── loss/               # Loss functions (MSE, MAE, CrossEntropy)
│       ├── math/               # Mathematical utilities
│       ├── models/             # ML models (supervised & unsupervised)
│       ├── neural/             # Neural network components
│       └── utils/              # Utility functions
├── test/                       # Test files
├── script/                     # Deployment scripts
└── foundry.toml               # Foundry configuration
```

### Setup untuk Foundry Project

1. **Clone repository atau tambahkan sebagai submodule:**

```bash
git clone https://github.com/rezacrown/Artemis.git
cd Artemis
```

2. **Install dependencies:**

```bash
forge install
```

3. **Build project:**

```bash
forge build
```

4. **Run tests:**

```bash
forge test
```

### Import dalam Contract

```solidity
// Import model yang diinginkan
import "Artemis/src/Artemis/models/supervised/LinearRegression.sol";
import "Artemis/src/Artemis/neural/NeuralNetwork.sol";
import "Artemis/src/Artemis/utils/DataPreprocessor.sol";
```

## 🎯 Quick Start

### Contoh 1: Linear Regression untuk House Price Prediction

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../Artemis/models/supervised/LinearRegression.sol";
import "../Artemis/utils/DataPreprocessor.sol";

contract HousePricePredictor {
    LinearRegression public model;

    constructor() {
        // Inisialisasi model dengan learning rate 0.01 dan regularization 0.001
        model = new LinearRegression(1e16, 1e15);
    }

    function trainModel() public returns (bool success) {
        uint256[][] memory features = getTrainingFeatures();
        uint256[] memory labels = getTrainingLabels();

        // Preprocess data
        (uint256[][] memory processedFeatures, uint256[] memory processedLabels, ) =
            DataPreprocessor.scaleFeatures(features, 0);

        // Train model dengan 100 epochs
        (success, ) = model.train(processedFeatures, processedLabels, 100, 1e16);
        return success;
    }

    function predictPrice(uint256 area, uint256 bedrooms) public view returns (uint256 price) {
        uint256[] memory features = new uint256[](2);
        features[0] = area * 1e18; // Convert ke fixed-point
        features[1] = bedrooms * 1e18;

        return model.predict(features);
    }
}
```

### Contoh 2: Neural Network untuk XOR Problem

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../Artemis/neural/NeuralNetwork.sol";
import "../Artemis/neural/layers/DenseLayer.sol";
import "../Artemis/activation/Sigmoid.sol";
import "../Artemis/loss/CrossEntropyLoss.sol";
import "../Artemis/neural/optimizers/AdamOptimizer.sol";

contract XORClassifier {
    NeuralNetwork public network;
    Sigmoid public sigmoid;
    CrossEntropyLoss public lossFunction;
    AdamOptimizer public optimizer;

    constructor() {
        sigmoid = new Sigmoid();
        lossFunction = new CrossEntropyLoss();
        optimizer = new AdamOptimizer();
        network = new NeuralNetwork(2, 1, lossFunction, optimizer);

        // Build network architecture
        network.addDenseLayer(4, true, sigmoid); // Hidden layer dengan 4 neurons
        network.addDenseLayer(1, true, sigmoid); // Output layer
    }

    function trainXOR() public returns (bool success) {
        uint256[][] memory features = getXORFeatures();
        uint256[] memory labels = getXORLabels();

        (success, ) = network.train(features, labels, 500, 1e17);
        return success;
    }

    function getXORFeatures() public pure returns (uint256[][] memory) {
        uint256[][] memory features = new uint256[][](4);
        uint256 constant ONE = 1e18;

        features[0] = new uint256[](2); // [0,0] -> 0
        features[0][0] = 0;
        features[0][1] = 0;

        features[1] = new uint256[](2); // [0,1] -> 1
        features[1][0] = 0;
        features[1][1] = ONE;

        features[2] = new uint256[](2); // [1,0] -> 1
        features[2][0] = ONE;
        features[2][1] = 0;

        features[3] = new uint256[](2); // [1,1] -> 0
        features[3][0] = ONE;
        features[3][1] = ONE;

        return features;
    }

    function getXORLabels() public pure returns (uint256[] memory) {
        uint256[] memory labels = new uint256[](4);
        uint256 constant ONE = 1e18;

        labels[0] = 0;  // [0,0] -> 0
        labels[1] = ONE; // [0,1] -> 1
        labels[2] = ONE; // [1,0] -> 1
        labels[3] = 0;   // [1,1] -> 0

        return labels;
    }
}
```

## 📚 API Documentation

### Core Interfaces

#### IModel Interface

```solidity
interface IModel {
    function train(uint256[][] features, uint256[] labels, uint256 epochs, uint256 learningRate)
        external returns (bool success, uint256 finalLoss);

    function predict(uint256[] features) external view returns (uint256 prediction);

    function evaluate(uint256[][] features, uint256[] labels)
        external view returns (uint256 accuracy, uint256 loss);

    function getTrainingStatus() external view returns (bool isTrained, uint256 epochs, uint256 loss);

    function getParameters() external view returns (uint256[] parameters);

    function setParameters(uint256[] parameters) external returns (bool success);
}
```

#### IActivation Interface

```solidity
interface IActivation {
    function activate(uint256 input) external pure returns (uint256 output);
    function derivative(uint256 input) external pure returns (uint256 derivative);
    function getActivationInfo() external pure returns (string name, string version, string type);
}
```

#### ILossFunction Interface

```solidity
interface ILossFunction {
    function calculateLoss(uint256 prediction, uint256 target) external pure returns (uint256 loss);
    function calculateGradient(uint256 prediction, uint256 target) external pure returns (int256 gradient);
    function getLossFunctionInfo() external pure returns (string name, string version, bool differentiable);
}
```

#### IOptimizer Interface

```solidity
interface IOptimizer {
    function updateParameters(uint256[] parameters, int256[] gradients, uint256 learningRate)
        external returns (uint256[] updatedParameters, uint256 updateMagnitude);

    function setLearningRate(uint256 newLearningRate) external returns (bool success);
    function getOptimizerInfo() external view returns (string name, string version, string type);
}
```

### Model-Specific Documentation

#### Linear Regression

**Use Cases:** Price prediction, trend analysis, continuous value prediction

**Parameters:**

- `learningRate`: Learning rate untuk gradient descent (recommended: 0.01-0.001)
- `regularization`: L2 regularization strength (recommended: 0.001-0.0001)

**Example:**

```solidity
LinearRegression model = new LinearRegression(1e16, 1e15); // 0.01 learning rate, 0.001 regularization
```

#### Logistic Regression

**Use Cases:** Binary classification, probability estimation

**Parameters:**

- `learningRate`: Learning rate untuk optimization
- `regularization`: L2 regularization untuk mencegah overfitting

#### K-Nearest Neighbors

**Use Cases:** Classification dan regression, pattern recognition

**Configuration:**

- `kValue`: Jumlah neighbors (recommended: 3-10)
- `distanceMetric`: Euclidean, Manhattan, atau Minkowski
- `weightingStrategy`: Uniform atau distance-based weighting

#### K-Means Clustering

**Use Cases:** Customer segmentation, data grouping, anomaly detection

**Parameters:**

- `numClusters`: Jumlah cluster (K)
- `maxIterations`: Maksimum iterasi training
- `initializationMethod`: Random, K-means++, atau Manual

#### Neural Network

**Use Cases:** Complex pattern recognition, non-linear relationships

**Architecture:**

- Multiple dense layers dengan activation functions
- Support untuk berbagai optimizers (SGD, Adam)
- Configurable loss functions

## 🔬 Examples

### TrainLinearRegression - House Price Prediction

**Dataset:** Sample house data dengan features (area, bedrooms) dan target (price)

**Features:**

- Area dalam square feet
- Number of bedrooms

**Target:** House price dalam USD

**Expected Results:**

- R-squared > 0.85 untuk dataset training
- Accurate price predictions untuk houses dengan features serupa

**Code:**

```solidity
// Lihat file lengkap di: src/Artemis/examples/TrainLinearRegression.sol
contract TrainLinearRegression {
    function runDemo() public returns (string memory results) {
        // Training, evaluation, dan prediction workflow lengkap
        return "Demo completed successfully";
    }
}
```

### TrainNeuralNetwork - XOR Classification

**Dataset:** XOR gate truth table

**Features:** [input1, input2] (0 atau 1)
**Target:** XOR output (0 atau 1)

**Architecture:**

- Input Layer: 2 nodes
- Hidden Layer: 4 nodes dengan Sigmoid activation
- Output Layer: 1 node dengan Sigmoid activation

**Expected Results:**

- Accuracy > 95% untuk XOR patterns
- Proper classification untuk semua input combinations

## ⚡ Gas Optimization Strategies

### Tips untuk Mengurangi Gas Costs

1. **Batch Processing**

```solidity
// Gunakan batch operations untuk mengurangi transaction count
function trainBatch(uint256[][] features, uint256[] labels, uint256 batchSize) public {
    for (uint256 i = 0; i < features.length; i += batchSize) {
        // Process batch
    }
}
```

2. **Storage Optimization**

- Gunakan `memory` daripada `storage` ketika memungkinkan
- Pack multiple variables ke dalam single storage slot
- Gunakan fixed-point arithmetic untuk efisiensi

3. **Model Complexity Management**

- Pilih model yang sesuai dengan complexity problem
- Gunakan regularization untuk mencegah overfitting
- Pertimbangkan trade-off antara accuracy dan gas costs

4. **Inference Optimization**

- Cache predictions ketika memungkinkan
- Gunakan simplified models untuk production
- Implement model compression techniques

### Gas Costs Estimates (Approximate)

| Operation                               | Gas Cost  | Complexity                 |
| --------------------------------------- | --------- | -------------------------- |
| Linear Regression Training (100 epochs) | ~500K gas | O(n×features×epochs)       |
| Neural Network Forward Pass             | ~50K gas  | O(layers×neurons)          |
| KNN Prediction                          | ~100K gas | O(n×features)              |
| K-Means Clustering (10 iterations)      | ~1M gas   | O(k×n×features×iterations) |

## ⚠️ Limitations and Considerations

### Constraints of On-Chain Machine Learning

1. **Computational Limits**

   - Gas limits membatasi complexity computations
   - Large datasets tidak praktis untuk on-chain processing
   - Iterative algorithms memerlukan careful gas management

2. **Numerical Precision**

   - Fixed-point arithmetic dengan precision 1e18
   - Potential untuk overflow/underflow
   - Limited numerical stability untuk complex operations

3. **Storage Costs**

   - Model parameters memerlukan storage space
   - Training data tidak praktis untuk disimpan on-chain
   - Consider off-chain storage dengan on-chain verification

4. **Recommended Use Cases**

   - ✅ Small to medium datasets
   - ✅ Simple to moderately complex models
   - ✅ Batch processing dengan reasonable sizes
   - ✅ Educational dan demonstrasi purposes

5. **Scenarios to Avoid**
   - ❌ Large-scale deep learning
   - ❌ Real-time continuous training
   - ❌ Very large datasets
   - ❌ High-frequency model updates

## 🛠️ Best Practices & Troubleshooting

### Model Selection Guide

| Problem Type               | Recommended Model   | Use Case                         | Gas Efficiency |
| -------------------------- | ------------------- | -------------------------------- | -------------- |
| Regression                 | Linear Regression   | Price prediction, trend analysis | ⭐⭐⭐⭐⭐     |
| Binary Classification      | Logistic Regression | Yes/No classification            | ⭐⭐⭐⭐       |
| Multi-class Classification | K-Nearest Neighbors | Pattern recognition              | ⭐⭐⭐         |
| Clustering                 | K-Means             | Customer segmentation            | ⭐⭐           |
| Complex Patterns           | Neural Network      | Non-linear relationships         | ⭐             |

### Common Issues & Solutions

#### Training Convergence Problems

**Issue:** Model loss tidak berkurang selama training
**Solutions:**

- Turunkan learning rate (0.01 → 0.001)
- Normalisasi features menggunakan DataPreprocessor
- Cek data quality dan remove outliers
- Increase number of training epochs

#### Overfitting

**Issue:** Training accuracy tinggi tapi test accuracy rendah
**Solutions:**

- Gunakan regularization (L2 untuk Linear/Logistic Regression)
- Kurangi model complexity
- Gunakan lebih banyak training data
- Implement early stopping

#### High Gas Costs

**Issue:** Training atau inference terlalu mahal
**Solutions:**

- Gunakan batch processing untuk large datasets
- Optimize model architecture (kurangi layers/neurons)
- Cache predictions untuk data yang sama
- Consider off-chain computation dengan on-chain verification

#### Numerical Stability

**Issue:** Overflow/underflow errors
**Solutions:**

- Gunakan fixed-point arithmetic dengan precision yang sesuai
- Normalisasi input data ke range [0, 1]
- Implement gradient clipping untuk neural networks
- Gunakan activation functions yang numerically stable (ReLU > Sigmoid)

### Performance Optimization Tips

1. **Data Preprocessing:**

   ```solidity
   // Selalu normalize data sebelum training
   (uint256[][] memory scaledFeatures, uint256[][] memory scalingParams) =
       DataPreprocessor.scaleFeatures(rawFeatures, 0); // 0 = min-max scaling
   ```

2. **Hyperparameter Tuning:**

   - Learning Rate: Start dengan 0.01, adjust berdasarkan convergence
   - Regularization: Gunakan 0.001 untuk mencegah overfitting
   - Batch Size: Gunakan batch processing untuk datasets besar

3. **Model Architecture:**
   - Start dengan model sederhana, tingkatkan complexity secara bertahap
   - Gunakan activation functions yang sesuai untuk problem type
   - Monitor training progress dengan events

## 🤝 Contributing

Kami menyambut kontribusi dari komunitas! Berikut guidelines untuk berkontribusi:

### Code Style Guidelines

- Gunakan Solidity ^0.8.0
- Ikuti [Solidity Style Guide](https://docs.soliditylang.org/en/latest/style-guide.html)
- Gunakan NatSpec comments untuk semua public functions
- Tulis comprehensive tests untuk semua new features

### Testing Requirements

```bash
# Run semua tests
forge test

# Run tests dengan gas reports
forge test --gas-report

# Run specific test file
forge test --match-path test/LinearRegression.t.sol
```

### Documentation Standards

- Update README.md untuk new features
- Tambahkan examples untuk demonstration
- Document gas costs dan performance characteristics
- Sertakan use cases dan best practices

### Pull Request Process

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

Artemis dilisensikan di bawah **MIT License** - lihat file [LICENSE](LICENSE) untuk detail lengkap.

```text
MIT License

Copyright (c) 2024 luxz71

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- Terinspirasi oleh scikit-learn dan framework machine learning modern
- Dibangun dengan [Foundry](https://getfoundry.sh/) toolkit
- Menggunakan fixed-point arithmetic untuk numerical stability
- Dirancang untuk educational purposes dan blockchain experimentation

---

**luxz71** - _Bringing Machine Learning to the Blockchain_
