
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../neural/NeuralNetwork.sol";
import "../neural/layers/DenseLayer.sol";
import "../neural/layers/ActivationLayer.sol";
import "../neural/optimizers/AdamOptimizer.sol";
import "../activation/Sigmoid.sol";
import "../loss/CrossEntropyLoss.sol";
import "../math/ArrayUtils.sol";

/**
 * @title TrainNeuralNetwork
 * @dev Contoh penggunaan Neural Network untuk XOR classification problem
 * @notice Demonstrasi lengkap neural network training dengan forward/backward propagation
 * @author Rizky Reza
 * 
 * @dev Use Case: XOR Problem Classification
 * - Dataset: XOR gate truth table
 * - Features: [input1, input2] (0 atau 1)
 * - Target: output (0 atau 1)
 * 
 * @dev Architecture: Neural Network dengan 2 input nodes, hidden layer, 1 output node
 * - Input Layer: 2 nodes
 * - Hidden Layer: 4 nodes dengan Sigmoid activation
 * - Output Layer: 1 node dengan Sigmoid activation
 * 
 * @dev Workflow:
 * 1. Network architecture configuration
 * 2. Layer addition dengan DenseLayer dan ActivationLayer
 * 3. Training dengan backpropagation dan Adam optimizer
 * 4. Evaluation dengan accuracy metrics
 * 5. Forward/backward propagation demonstration
 * 
 * @dev Features Demonstrated:
 * - Multi-layer neural network construction
 * - Forward propagation untuk inference
 * - Backward propagation untuk training
 * - Optimizer usage dengan Adam
 * - Activation functions dengan Sigmoid
 * - Loss function dengan Cross Entropy
 * - Training progress monitoring
 */
contract TrainNeuralNetwork {
    using ArrayUtils for uint256[];
    
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    
    // Network components
    NeuralNetwork public network;
    AdamOptimizer public optimizer;
    CrossEntropyLoss public lossFunction;
    Sigmoid public sigmoidActivation;
    
    // Training state
    bool public isTrained;
    uint256 public trainingEpochs;
    uint256 public finalLoss;
    uint256 public finalAccuracy;
    
    // Network architecture
    uint256 public inputSize;
    uint256 public hiddenSize;
    uint256 public outputSize;
    
    // Events untuk monitoring
    event NetworkInitialized(address networkAddress, uint256 inputSize, uint256 hiddenSize, uint256 outputSize);
    event LayerAdded(string layerType, uint256 layerIndex, uint256 size);
    event TrainingProgress(uint256 epoch, uint256 loss, uint256 accuracy);
    event TrainingCompleted(uint256 finalLoss, uint256 finalAccuracy, uint256 trainingTime);
    event PredictionMade(uint256[] input, uint256 predictedOutput, uint256 actualOutput);
    event NetworkEvaluated(uint256 testSize, uint256 accuracy, uint256 loss);
    
    /**
     * @dev Constructor untuk inisialisasi neural network
     * @param _hiddenSize Ukuran hidden layer
     * @param learningRate Learning rate untuk training
     */
    constructor(uint256 _hiddenSize, uint256 learningRate) {
        inputSize = 2;  // XOR memiliki 2 input
        hiddenSize = _hiddenSize;
        outputSize = 1; // Binary classification output
        
        // Inisialisasi komponen neural network
        sigmoidActivation = new Sigmoid();
        lossFunction = new CrossEntropyLoss();
        optimizer = new AdamOptimizer();
        
        // Inisialisasi neural network
        network = new NeuralNetwork(
            inputSize,
            outputSize,
            lossFunction,
            optimizer
        );
        
        // Build network architecture
        _buildNetworkArchitecture();
        
        // Initialize training state
        isTrained = false;
        trainingEpochs = 0;
        finalLoss = 0;
        finalAccuracy = 0;
        
        emit NetworkInitialized(address(network), inputSize, hiddenSize, outputSize);
    }
    
    /**
     * @dev Membangun architecture neural network
     * @dev Architecture: Input(2) -> Dense(4) -> Sigmoid -> Dense(1) -> Sigmoid
     */
    function _buildNetworkArchitecture() internal {
        // Tambahkan hidden layer: Dense layer dengan 4 neurons
        network.addDenseLayer(
            hiddenSize,         // 4 neurons
            true,               // use bias
            sigmoidActivation   // Sigmoid activation
        );
        
        emit LayerAdded("Dense", 0, hiddenSize);
        
        // Tambahkan output layer: Dense layer dengan 1 neuron
        network.addDenseLayer(
            outputSize,         // 1 neuron
            true,               // use bias
            sigmoidActivation   // Sigmoid activation
        );
        
        emit LayerAdded("Dense", 1, outputSize);
    }
    
    /**
     * @dev Mendapatkan XOR dataset untuk training
     * @return features Array 2D features: [input1, input2]
     * @return labels Array 1D labels: XOR output (0 atau 1)
     * 
     * @dev XOR Truth Table:
     * [0, 0] -> 0
     * [0, 1] -> 1
     * [1, 0] -> 1
     * [1, 1] -> 0
     */
    function getXORDataset() public pure returns (
        uint256[][] memory features,
        uint256[] memory labels
    ) {
        features = new uint256[][](4);
        labels = new uint256[](4);
        
        // XOR pattern 1: [0, 0] -> 0
        features[0] = new uint256[](2);
        features[0][0] = 0;
        features[0][1] = 0;
        labels[0] = 0;
        
        // XOR pattern 2: [0, 1] -> 1
        features[1] = new uint256[](2);
        features[1][0] = 0;
        features[1][1] = ONE;
        labels[1] = ONE;
        
        // XOR pattern 3: [1, 0] -> 1
        features[2] = new uint256[](2);
        features[2][0] = ONE;
        features[2][1] = 0;
        labels[2] = ONE;
        
        // XOR pattern 4: [1, 1] -> 0
        features[3] = new uint256[](2);
        features[3][0] = ONE;
        features[3][1] = ONE;
        labels[3] = 0;
        
        return (features, labels);
    }
    
    /**
     * @dev Mendapatkan test dataset untuk evaluation
     * @return features Array 2D test features
     * @return labels Array 1D test labels
     */
    function getTestDataset() public pure returns (
        uint256[][] memory features,
        uint256[] memory labels
    ) {
        features = new uint256[][](2);
        labels = new uint256[](2);
        
        // Test pattern 1: [0.5, 0.5] -> expected ~0
        features[0] = new uint256[](2);
        features[0][0] = ONE / 2;
        features[0][1] = ONE / 2;
        labels[0] = 0;
        
        // Test pattern 2: [0.5, 0] -> expected ~1
        features[1] = new uint256[](2);
        features[1][0] = ONE / 2;
        features[1][1] = 0;
        labels[1] = ONE;
        
        return (features, labels);
    }
    
    /**
     * @dev Melakukan training neural network dengan XOR dataset
     * @param epochs Jumlah epoch training
     * @param learningRate Learning rate untuk training
     * @return success Status keberhasilan training
     * @return trainingLoss Nilai loss akhir setelah training
     * 
     * @dev Training Process:
     * 1. Load XOR dataset
     * 2. Train network dengan backpropagation
     * 3. Monitor progress dengan events
     * 4. Simpan training statistics
     */
    function trainModel(uint256 epochs, uint256 learningRate) public returns (bool success, uint256 trainingLoss) {
        require(epochs > 0, "TrainNeuralNetwork: Epochs must be positive");
        require(learningRate > 0, "TrainNeuralNetwork: Learning rate must be positive");
        
        // Dapatkan XOR dataset
        (uint256[][] memory features, uint256[] memory labels) = getXORDataset();
        
        // Train network
        uint256 startTime = block.timestamp;
        (success, trainingLoss) = network.train(features, labels, epochs, learningRate);
        
        if (success) {
            isTrained = true;
            trainingEpochs = epochs;
            finalLoss = trainingLoss;
            
            // Evaluate model untuk mendapatkan accuracy
            (uint256 accuracy, ) = network.evaluate(features, labels);
            finalAccuracy = accuracy;
            
            uint256 trainingTime = block.timestamp - startTime;
            
            emit TrainingCompleted(trainingLoss, finalAccuracy, trainingTime);
        }
        
        return (success, trainingLoss);
    }
    
    /**
     * @dev Melakukan prediksi XOR output berdasarkan input
     * @param input1 Input pertama (0 atau 1 dalam fixed-point)
     * @param input2 Input kedua (0 atau 1 dalam fixed-point)
     * @return predictedOutput Prediksi output (0-1 dalam fixed-point)
     *
     * @dev Prediction Process:
     * 1. Prepare input features
     * 2. Lakukan forward propagation melalui network
     * 3. Return probability output
     */
    function predictXOR(uint256 input1, uint256 input2) public returns (uint256 predictedOutput) {
        require(isTrained, "TrainNeuralNetwork: Network not trained");
        
        // Prepare input features
        uint256[] memory features = new uint256[](2);
        features[0] = input1;
        features[1] = input2;
        
        // Lakukan prediksi
        predictedOutput = network.predict(features);
        
        // Emit prediction event
        uint256 actualOutput = _getExpectedXOROutput(input1, input2);
        emit PredictionMade(features, predictedOutput, actualOutput);
        
        return predictedOutput;
    }
    
    /**
     * @dev Mendapatkan expected XOR output berdasarkan input
     * @param input1 Input pertama
     * @param input2 Input kedua
     * @return expectedOutput Expected XOR output
     */
    function _getExpectedXOROutput(uint256 input1, uint256 input2) internal pure returns (uint256 expectedOutput) {
        // Threshold untuk menentukan binary values
        uint256 threshold = ONE / 2;
        
        bool bin1 = input1 > threshold;
        bool bin2 = input2 > threshold;
        
        // XOR operation
        if (bin1 != bin2) {
            return ONE; // 1
        } else {
            return 0;   // 0
        }
    }
    
    /**
     * @dev Mengevaluasi performa neural network dengan test dataset
     * @return accuracy Classification accuracy
     * @return loss Cross entropy loss
     *
     * @dev Evaluation Metrics:
     * - Accuracy: Proporsi prediksi yang benar
     * - Cross Entropy Loss: Measure of prediction uncertainty
     */
    function evaluateModel() public returns (uint256 accuracy, uint256 loss) {
        require(isTrained, "TrainNeuralNetwork: Network not trained");
        
        (uint256[][] memory testFeatures, uint256[] memory testLabels) = getTestDataset();
        
        // Evaluate network
        (accuracy, loss) = network.evaluate(testFeatures, testLabels);
        
        emit NetworkEvaluated(testFeatures.length, accuracy, loss);
        
        return (accuracy, loss);
    }
    
    /**
     * @dev Mendapatkan informasi tentang network architecture
     * @return layerCount Jumlah layers
     * @return totalParameters Total jumlah parameters
     * @return layerInfo Array informasi setiap layer
     */
    function getNetworkArchitecture() public view returns (
        uint256 layerCount,
        uint256 totalParameters,
        string[] memory layerInfo
    ) {
        return network.getNetworkArchitecture();
    }
    
    /**
     * @dev Mendapatkan training history
     * @return lossHistory Array loss per epoch
     * @return accuracyHistory Array accuracy per epoch
     */
    function getTrainingHistory() public view returns (
        uint256[] memory lossHistory,
        uint256[] memory accuracyHistory
    ) {
        return network.getTrainingHistory();
    }
    
    /**
     * @dev Mendapatkan informasi lengkap tentang neural network
     * @return networkInfo Metadata network
     * @return trainingInfo Status training
     * @return architectureInfo Informasi architecture
     */
    function getNetworkInfo() public view returns (
        string memory networkInfo,
        string memory trainingInfo,
        string memory architectureInfo
    ) {
        (string memory name, string memory version, uint256 inputSize, uint256 outputSize) = network.getModelInfo();
        (bool trained, uint256 epochs, uint256 loss) = network.getTrainingStatus();
        (uint256 layerCount, uint256 totalParams, ) = getNetworkArchitecture();
        
        networkInfo = string(abi.encodePacked(
            "Network: ", name, " v", version, 
            " | Input: ", _uintToString(inputSize),
            " | Output: ", _uintToString(outputSize)
        ));
        
        trainingInfo = string(abi.encodePacked(
            "Trained: ", trained ? "Yes" : "No",
            " | Epochs: ", _uintToString(epochs),
            " | Loss: ", _uintToString(loss),
            " | Accuracy: ", _uintToString(finalAccuracy)
        ));
        
        architectureInfo = string(abi.encodePacked(
            "Layers: ", _uintToString(layerCount),
            " | Parameters: ", _uintToString(totalParams),
            " | Architecture: Input(", _uintToString(inputSize), ")->Hidden(", _uintToString(hiddenSize), ")->Output(", _uintToString(outputSize), ")"
        ));
        
        return (networkInfo, trainingInfo, architectureInfo);
    }
    
    /**
     * @dev Demo function untuk menunjukkan neural network workflow lengkap
     * @return demoResults Hasil demonstrasi
     */
    function runDemo() public returns (string memory demoResults) {
        // Step 1: Train network
        (bool success, uint256 loss) = trainModel(500, (1 * PRECISION) / 10); // 500 epochs, 0.1 learning rate
        
        if (!success) {
            return "Demo failed: Training unsuccessful";
        }
        
        // Step 2: Evaluate network
        (uint256 accuracy, uint256 testLoss) = evaluateModel();
        
        // Step 3: Make predictions untuk semua XOR patterns
        uint256 prediction00 = predictXOR(0, 0);
        uint256 prediction01 = predictXOR(0, ONE);
        uint256 prediction10 = predictXOR(ONE, 0);
        uint256 prediction11 = predictXOR(ONE, ONE);
        
        // Step 4: Get network architecture info
        (uint256 layerCount, uint256 totalParams, ) = getNetworkArchitecture();
        
        demoResults = string(abi.encodePacked(
            "=== Artemis Neural Network Demo (XOR Problem) ===\n",
            "Training: Success (Loss: ", _uintToString(loss), ")\n",
            "Evaluation: Accuracy=", _uintToString(accuracy), ", Test Loss=", _uintToString(testLoss), "\n",
            "XOR Predictions:\n",
            "  [0,0] -> ", _formatProbability(prediction00), " (expected: 0)\n",
            "  [0,1] -> ", _formatProbability(prediction01), " (expected: 1)\n", 
            "  [1,0] -> ", _formatProbability(prediction10), " (expected: 1)\n",
            "  [1,1] -> ", _formatProbability(prediction11), " (expected: 0)\n",
            "Network: ", _uintToString(layerCount), " layers, ", _uintToString(totalParams), " parameters"
        ));
        
        return demoResults;
    }
    
    /**
     * @dev Format probability untuk display yang lebih readable
     * @param probability Nilai probability
     * @return formatted String probability yang sudah diformat
     */
    function _formatProbability(uint256 probability) internal pure returns (string memory formatted) {
        uint256 percentage = (probability * 100) / PRECISION;
        return string(abi.encodePacked(_uintToString(percentage), "%"));
    }
    
    /**
     * @dev Fungsi internal untuk konversi uint ke string
     * @param _i Nilai uint
     * @return _uintAsString String hasil konversi
     */
    function _uintToString(uint256 _i) internal pure returns (string memory _uintAsString) {
        if (_i == 0) {
            return "0";
        }
        uint256 j = _i;
        uint256 len;
        while (j != 0) {
            len++;
            j /= 10;
        }
        bytes memory bstr = new bytes(len);
        uint256 k = len;
        while (_i != 0) {
            k = k - 1;
            uint8 temp = (48 + uint8(_i - (_i / 10) * 10));
            bytes1 b1 = bytes1(temp);
            bstr[k] = b1;
            _i /= 10;
        }
        return string(bstr);
    }
    
    /**
     * @dev Mendapatkan best practices untuk Neural Networks
     * @return practices Tips dan best practices
     */
    function getBestPractices() public pure returns (string memory practices) {
        practices = string(abi.encodePacked(
            "=== Neural Network Best Practices ===\n",
            "1. Architecture Design: Mulai dengan architecture sederhana, tingkatkan complexity secara bertahap\n",
            "2. Activation Functions: Gunakan Sigmoid/ReLU untuk hidden layers, Sigmoid untuk binary classification\n",
            "3. Weight Initialization",
            "4. Learning Rate: Gunakan learning rate yang sesuai (0.1-0.001) dan pertimbangkan learning rate scheduling\n",
            "5. Regularization: Gunakan L2 regularization atau dropout untuk mencegah overfitting\n",
            "6. Batch Training: Untuk dataset besar, gunakan mini-batch training untuk efisiensi\n",
            "7. Monitoring: Pantau training loss dan validation accuracy untuk deteksi overfitting"
        ));
        return practices;
    }
    
    /**
     * @dev Mendapatkan troubleshooting guide untuk neural networks
     * @return troubleshooting Panduan troubleshooting
     */
    function getTroubleshootingGuide() public pure returns (string memory troubleshooting) {
        troubleshooting = string(abi.encodePacked(
            "=== Neural Network Troubleshooting Guide ===\n",
            "Issue: Network tidak belajar (loss tidak berkurang)\n",
            "Solution: Cek learning rate, initialization weights, network architecture\n\n",
            "Issue: Overfitting (training loss rendah, test loss tinggi)\n",
            "Solution: Tambahkan regularization, kurangi network complexity, gunakan lebih banyak data\n\n",
            "Issue: Vanishing/Exploding gradients\n",
            "Solution: Gunakan activation functions yang sesuai (ReLU), normalisasi input, gradient clipping\n\n",
            "Issue: Training sangat lambat\n",
            "Solution: Optimize batch size, pertimbangkan distributed training untuk dataset besar\n\n",
            "Issue: Predictions tidak konsisten\n",
            "Solution: Cek data preprocessing, pastikan konsistensi antara training dan inference"
        ));
        return troubleshooting;
    }
    
    /**
     * @dev Mendemonstrasikan forward propagation manual untuk educational purposes
     * @param input1 Input pertama
     * @param input2 Input kedua
     * @return hiddenLayerOutput Output dari hidden layer
     * @return finalOutput Output final dari network
     *
     * @dev Educational Purpose:
     * Demonstrasi bagaimana forward propagation bekerja layer oleh layer
     */
    function demonstrateForwardPropagation(uint256 input1, uint256 input2) public returns (
        uint256[] memory hiddenLayerOutput,
        uint256 finalOutput
    ) {
        require(isTrained, "TrainNeuralNetwork: Network not trained");
        
        // Untuk simplicity, kita akan return array kosong dan prediksi normal
        // Dalam implementasi produksi, kita akan mengakses internal layer states
        hiddenLayerOutput = new uint256[](0);
        finalOutput = predictXOR(input1, input2);
        
        return (hiddenLayerOutput, finalOutput);
    }
}