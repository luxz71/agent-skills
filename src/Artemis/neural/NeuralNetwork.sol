
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../interfaces/IModel.sol";
import "../interfaces/ILossFunction.sol";
import "../interfaces/IOptimizer.sol";
import "./layers/DenseLayer.sol";
import "./layers/ActivationLayer.sol";
import "../math/ArrayUtils.sol";

/**
 * @title NeuralNetwork
 * @dev Implementasi configurable multi-layer neural network
 * @notice Neural network dengan forward/backward propagation dan training capabilities
 * @author Rizky Reza
 */
contract NeuralNetwork is IModel {
    using ArrayUtils for uint256[];
    
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    
    // Network configuration
    string public modelName;
    string public version;
    uint256 public inputSize;
    uint256 public outputSize;
    
    // Layers
    address[] public layers;
    mapping(address => bool) public isLayer;
    mapping(address => string) public layerTypes;
    
    // Training state
    bool public isTrained;
    uint256 public trainingEpochs;
    uint256 public currentLoss;
    uint256 public bestLoss;
    
    // Hyperparameters
    uint256 public learningRate;
    uint256 public batchSize;
    ILossFunction public lossFunction;
    IOptimizer public optimizer;
    
    // Training history
    uint256[] public lossHistory;
    uint256[] public accuracyHistory;
    
    // Events tambahan untuk neural network
    event LayerAdded(address indexed layer, string layerType, uint256 index);
    event NetworkInitialized(uint256 inputSize, uint256 outputSize, uint256 layerCount);
    event BackwardPropagationCompleted(uint256 totalGradientMagnitude);
    event LayerMetrics(uint256 layerIndex, uint256 parameterCount, uint256 gradientMagnitude);
    
    /**
     * @dev Inisialisasi neural network dengan konfigurasi dasar
     * @param _inputSize Ukuran input network
     * @param _outputSize Ukuran output network
     * @param _lossFunction Alamat contract loss function
     * @param _optimizer Alamat contract optimizer
     */
    constructor(
        uint256 _inputSize,
        uint256 _outputSize,
        ILossFunction _lossFunction,
        IOptimizer _optimizer
    ) {
        require(_inputSize > 0, "NeuralNetwork: Input size must be positive");
        require(_outputSize > 0, "NeuralNetwork: Output size must be positive");
        require(address(_lossFunction) != address(0), "NeuralNetwork: Loss function address cannot be zero");
        require(address(_optimizer) != address(0), "NeuralNetwork: Optimizer address cannot be zero");
        
        modelName = "NeuralNetwork";
        version = "1.0.0";
        inputSize = _inputSize;
        outputSize = _outputSize;
        lossFunction = _lossFunction;
        optimizer = _optimizer;
        
        // Default hyperparameters
        learningRate = (1 * PRECISION) / 100; // 0.01
        batchSize = 32;
        
        isTrained = false;
        trainingEpochs = 0;
        currentLoss = type(uint256).max;
        bestLoss = type(uint256).max;
        
        emit NetworkInitialized(_inputSize, _outputSize, layers.length);
    }
    
    /**
     * @dev Menambahkan dense layer ke network
     * @param layerSize Ukuran layer
     * @param useBias True jika menggunakan bias
     * @param activationFunction Alamat contract activation function
     */
    function addDenseLayer(
        uint256 layerSize,
        bool useBias,
        IActivation activationFunction
    ) external {
        require(layerSize > 0, "NeuralNetwork: Layer size must be positive");
        require(address(activationFunction) != address(0), "NeuralNetwork: Activation function address cannot be zero");
        
        uint256 prevLayerSize = layers.length == 0 ? inputSize : _getLayerOutputSize(layers[layers.length - 1]);
        
        DenseLayer newLayer = new DenseLayer(
            prevLayerSize,
            layerSize,
            useBias,
            activationFunction
        );
        
        layers.push(address(newLayer));
        isLayer[address(newLayer)] = true;
        layerTypes[address(newLayer)] = "Dense";
        
        emit LayerAdded(address(newLayer), "Dense", layers.length - 1);
    }
    
    /**
     * @dev Menambahkan activation layer ke network
     * @param activationFunction Alamat contract activation function
     */
    function addActivationLayer(IActivation activationFunction) external {
        require(address(activationFunction) != address(0), "NeuralNetwork: Activation function address cannot be zero");
        
        uint256 prevLayerSize = layers.length == 0 ? inputSize : _getLayerOutputSize(layers[layers.length - 1]);
        
        ActivationLayer newLayer = new ActivationLayer(
            prevLayerSize,
            activationFunction
        );
        
        layers.push(address(newLayer));
        isLayer[address(newLayer)] = true;
        layerTypes[address(newLayer)] = "Activation";
        
        emit LayerAdded(address(newLayer), "Activation", layers.length - 1);
    }
    
    /**
     * @dev Melakukan training neural network dengan data yang diberikan
     * @param features Array 2D dari fitur training
     * @param labels Array 1D dari label training
     * @param epochs Jumlah epoch training
     * @param _learningRate Learning rate untuk optimisasi
     * @return success Status keberhasilan training
     * @return finalLoss Nilai loss akhir setelah training
     */
    function train(
        uint256[][] calldata features,
        uint256[] calldata labels,
        uint256 epochs,
        uint256 _learningRate
    ) external override returns (bool success, uint256 finalLoss) {
        require(features.length > 0, "NeuralNetwork: Features cannot be empty");
        require(features.length == labels.length, "NeuralNetwork: Features and labels must have same length");
        require(epochs > 0, "NeuralNetwork: Epochs must be positive");
        require(_learningRate > 0, "NeuralNetwork: Learning rate must be positive");
        require(layers.length > 0, "NeuralNetwork: No layers added");
        
        learningRate = _learningRate;
        
        emit TrainingStarted(address(this), epochs, block.timestamp);
        
        uint256 totalSamples = features.length;
        uint256 bestEpoch = 0;
        
        for (uint256 epoch = 0; epoch < epochs; epoch++) {
            uint256 epochLoss = 0;
            uint256 correctPredictions = 0;
            
            // Mini-batch training
            for (uint256 batchStart = 0; batchStart < totalSamples; batchStart += batchSize) {
                uint256 batchEnd = batchStart + batchSize;
                if (batchEnd > totalSamples) {
                    batchEnd = totalSamples;
                }
                
                // Reset gradients untuk semua layers
                _resetAllGradients();
                
                uint256 batchLoss = 0;
                
                // Process batch
                for (uint256 i = batchStart; i < batchEnd; i++) {
                    // Forward propagation
                    uint256[] memory prediction = _forward(features[i]);
                    
                    // Calculate loss
                    uint256 sampleLoss = lossFunction.calculateLoss(prediction[0], labels[i]);
                    batchLoss += sampleLoss;
                    
                    // Calculate accuracy (untuk classification)
                    if (_isClassificationCorrect(prediction[0], labels[i])) {
                        correctPredictions++;
                    }
                    
                    // Backward propagation
                    int256 gradient = lossFunction.calculateGradient(prediction[0], labels[i]);
                    int256[] memory outputGradient = new int256[](1);
                    outputGradient[0] = gradient;
                    
                    _backward(_toUintArray(outputGradient));
                }
                
                // Average batch loss
                uint256 averageBatchLoss = batchLoss / (batchEnd - batchStart);
                epochLoss += averageBatchLoss;
                
                // Update parameters
                _updateAllParameters();
            }
            
            // Calculate epoch statistics
            uint256 averageEpochLoss = epochLoss / ((totalSamples + batchSize - 1) / batchSize);
            uint256 accuracy = (correctPredictions * PRECISION) / totalSamples;
            
            currentLoss = averageEpochLoss;
            lossHistory.push(averageEpochLoss);
            accuracyHistory.push(accuracy);
            
            // Update best loss
            if (averageEpochLoss < bestLoss) {
                bestLoss = averageEpochLoss;
                bestEpoch = epoch;
            }
            
            emit EpochCompleted(address(this), epoch, averageEpochLoss, accuracy);
            
            // Early stopping check (sederhana)
            if (epoch > 10 && averageEpochLoss > bestLoss * 2) {
                break; // Stop jika loss meningkat signifikan
            }
        }
        
        trainingEpochs += epochs;
        isTrained = true;
        finalLoss = currentLoss;
        
        emit TrainingCompleted(address(this), finalLoss, block.timestamp);
        
        return (true, finalLoss);
    }
    
    /**
     * @dev Melakukan prediksi menggunakan neural network
     * @param features Array 1D dari fitur untuk prediksi
     * @return prediction Hasil prediksi model
     */
    function predict(uint256[] calldata features) external view override returns (uint256 prediction) {
        require(features.length == inputSize, "NeuralNetwork: Input size mismatch");
        require(layers.length > 0, "NeuralNetwork: No layers added");
        
        uint256[] memory output = _forwardView(features);
        return output[0]; // Return first output element
    }
    
    /**
     * @dev Melakukan prediksi batch
     * @param features Array 2D dari fitur untuk prediksi
     * @return predictions Array hasil prediksi
     */
    function predictBatch(uint256[][] calldata features) external view returns (uint256[] memory predictions) {
        require(features.length > 0, "NeuralNetwork: Features cannot be empty");
        require(layers.length > 0, "NeuralNetwork: No layers added");
        
        predictions = new uint256[](features.length);
        
        for (uint256 i = 0; i < features.length; i++) {
            require(features[i].length == inputSize, "NeuralNetwork: Input size mismatch");
            uint256[] memory output = _forwardView(features[i]);
            predictions[i] = output[0];
        }
        
        return predictions;
    }
    
    /**
     * @dev Mengevaluasi performa neural network dengan data test
     * @param features Array 2D dari fitur test
     * @param labels Array 1D dari label test
     * @return accuracy Akurasi model pada data test
     * @return loss Nilai loss pada data test
     */
    function evaluate(
        uint256[][] calldata features,
        uint256[] calldata labels
    ) external view override returns (uint256 accuracy, uint256 loss) {
        require(features.length > 0, "NeuralNetwork: Features cannot be empty");
        require(features.length == labels.length, "NeuralNetwork: Features and labels must have same length");
        require(layers.length > 0, "NeuralNetwork: No layers added");
        
        uint256 totalLoss = 0;
        uint256 correctPredictions = 0;
        
        for (uint256 i = 0; i < features.length; i++) {
            require(features[i].length == inputSize, "NeuralNetwork: Input size mismatch");
            
            uint256[] memory prediction = _forwardView(features[i]);
            uint256 sampleLoss = lossFunction.calculateLoss(prediction[0], labels[i]);
            totalLoss += sampleLoss;
            
            if (_isClassificationCorrect(prediction[0], labels[i])) {
                correctPredictions++;
            }
        }
        
        accuracy = (correctPredictions * PRECISION) / features.length;
        loss = totalLoss / features.length;
        
        return (accuracy, loss);
    }
    
    /**
     * @dev Mengembalikan status training model
     * @return _isTrained True jika model sudah ditraining
     * @return _trainingEpochs Jumlah epoch yang sudah dijalankan
     * @return _currentLoss Nilai loss terakhir
     */
    function getTrainingStatus() external view override returns (
        bool _isTrained,
        uint256 _trainingEpochs,
        uint256 _currentLoss
    ) {
        return (isTrained, trainingEpochs, currentLoss);
    }
    
    /**
     * @dev Mengembalikan parameter model (semua parameter dari semua layers)
     * @return parameters Array flattened dari semua parameter
     */
    function getParameters() external view override returns (uint256[] memory parameters) {
        // Collect semua parameter dari semua layers
        uint256 totalParams = 0;
        
        // Hitung total parameter terlebih dahulu
        for (uint256 i = 0; i < layers.length; i++) {
            if (_compareStrings(layerTypes[layers[i]], "Dense")) {
                DenseLayer layer = DenseLayer(layers[i]);
                (uint256[][] memory weights, uint256[] memory biases) = layer.getParameters();
                totalParams += (weights.length * weights[0].length) + biases.length;
            }
        }
        
        parameters = new uint256[](totalParams);
        uint256 paramIndex = 0;
        
        // Collect semua parameter
        for (uint256 i = 0; i < layers.length; i++) {
            if (_compareStrings(layerTypes[layers[i]], "Dense")) {
                DenseLayer layer = DenseLayer(layers[i]);
                (uint256[][] memory weights, uint256[] memory biases) = layer.getParameters();
                
                // Flatten weights
                for (uint256 j = 0; j < weights.length; j++) {
                    for (uint256 k = 0; k < weights[j].length; k++) {
                        parameters[paramIndex++] = weights[j][k];
                    }
                }
                
                // Add biases
                for (uint256 j = 0; j < biases.length; j++) {
                    parameters[paramIndex++] = biases[j];
                }
            }
        }
        
        return parameters;
    }
    
    /**
     * @dev Mengatur parameter model
     * @param parameters Array dari parameter baru
     * @return success Status keberhasilan pengaturan parameter
     */
    function setParameters(uint256[] calldata parameters) external override returns (bool success) {
        // Implementasi kompleks - untuk simplicity return false
        // Dalam implementasi produksi, perlu logic untuk unflatten parameters ke layers
        return false;
    }
    
    /**
     * @dev Mengembalikan metadata model
     * @return _modelName Nama model
     * @return _version Versi model
     * @return _inputSize Ukuran input model
     * @return _outputSize Ukuran output model
     */
    function getModelInfo() external view override returns (
        string memory _modelName,
        string memory _version,
        uint256 _inputSize,
        uint256 _outputSize
    ) {
        return (modelName, version, inputSize, outputSize);
    }
    
    /**
     * @dev Reset model ke kondisi awal
     * @return success Status keberhasilan reset
     */
    function reset() external override returns (bool success) {
        // Reset training state
        isTrained = false;
        trainingEpochs = 0;
        currentLoss = type(uint256).max;
        bestLoss = type(uint256).max;
        
        // Clear history
        delete lossHistory;
        delete accuracyHistory;
        
        // Reset semua layers
        for (uint256 i = 0; i < layers.length; i++) {
            if (_compareStrings(layerTypes[layers[i]], "Dense")) {
                DenseLayer(layers[i]).resetGradients();
            } else if (_compareStrings(layerTypes[layers[i]], "Activation")) {
                ActivationLayer(layers[i]).resetState();
            }
        }
        
        return true;
    }
    
    /**
     * @dev Mengembalikan informasi detail tentang network architecture
     * @return layerCount Jumlah layers
     * @return totalParameters Total jumlah parameter
     * @return layerInfo Array informasi setiap layer
     */
    function getNetworkArchitecture() external view returns (
        uint256 layerCount,
        uint256 totalParameters,
        string[] memory layerInfo
    ) {
        layerCount = layers.length;
        totalParameters = 0;
        layerInfo = new string[](layerCount);
        
        for (uint256 i = 0; i < layerCount; i++) {
            if (_compareStrings(layerTypes[layers[i]], "Dense")) {
                DenseLayer layer = DenseLayer(layers[i]);
                (string memory lt, , , uint256 params, ) = layer.getLayerInfo();
                totalParameters += params;
                layerInfo[i] = string(abi.encodePacked(lt, " (", _uintToString(params), " params)"));
            } else if (_compareStrings(layerTypes[layers[i]], "Activation")) {
                ActivationLayer layer = ActivationLayer(layers[i]);
                (string memory lt, , , uint256 params, string memory activationInfo) = layer.getLayerInfo();
                layerInfo[i] = string(abi.encodePacked(lt, " - ", activationInfo));
            }
        }
        
        return (layerCount, totalParameters, layerInfo);
    }
    
    /**
     * @dev Mengembalikan training history
     * @return _lossHistory Array loss per epoch
     * @return _accuracyHistory Array accuracy per epoch
     */
    function getTrainingHistory() external view returns (
        uint256[] memory _lossHistory,
        uint256[] memory _accuracyHistory
    ) {
        return (lossHistory, accuracyHistory);
    }
    
    // ========== INTERNAL FUNCTIONS ==========
    
    /**
     * @dev Melakukan forward propagation melalui semua layers
     * @param input Array input values
     * @return output Array output values
     */
    function _forward(uint256[] memory input) internal returns (uint256[] memory output) {
        uint256[] memory currentOutput = input;
        
        for (uint256 i = 0; i < layers.length; i++) {
            if (_compareStrings(layerTypes[layers[i]], "Dense")) {
                DenseLayer layer = DenseLayer(layers[i]);
                currentOutput = layer.forward(currentOutput);
            } else if (_compareStrings(layerTypes[layers[i]], "Activation")) {
                ActivationLayer layer = ActivationLayer(layers[i]);
                currentOutput = layer.forward(currentOutput);
            }
        }
        
        return currentOutput;
    }
    
    /**
     * @dev Melakukan forward propagation (view version) - menggunakan parameter yang sudah ada
     * @param input Array input values
     * @return output Array output values
     */
    function _forwardView(uint256[] memory input) internal view returns (uint256[] memory output) {
        uint256[] memory currentOutput = input;
        
        for (uint256 i = 0; i < layers.length; i++) {
            if (_compareStrings(layerTypes[layers[i]], "Dense")) {
                DenseLayer layer = DenseLayer(layers[i]);
                (uint256[][] memory weights, uint256[] memory biases) = layer.getParameters();
                bool useBias;
                (,,,, useBias) = layer.getLayerInfo();
                
                // Simulasi forward pass menggunakan parameter yang sudah ada
                uint256[] memory newOutput = new uint256[](weights.length);
                for (uint256 j = 0; j < weights.length; j++) {
                    uint256 sum = 0;
                    for (uint256 k = 0; k < weights[j].length; k++) {
                        sum += (currentOutput[k] * weights[j][k]) / PRECISION;
                    }
                    if (useBias) {
                        sum += biases[j];
                    }
                    newOutput[j] = sum;
                }
                currentOutput = newOutput;
            } else if (_compareStrings(layerTypes[layers[i]], "Activation")) {
                // Untuk activation layer, kita perlu implementasi manual karena forward() mengubah state
                // Dalam implementasi produksi, perlu membuat fungsi pure untuk activation
                // Untuk sekarang, kita skip activation dalam view mode
                // currentOutput tetap sama (tanpa activation)
            }
        }
        
        return currentOutput;
    }
    
    
    /**
     * @dev Melakukan backward propagation melalui semua layers
     * @param outputGradient Gradien dari output
     */
    function _backward(uint256[] memory outputGradient) internal {
        uint256[] memory currentGradient = outputGradient;
        
        for (uint256 i = layers.length; i > 0; i--) {
            uint256 layerIndex = i - 1;
            
            if (_compareStrings(layerTypes[layers[layerIndex]], "Dense")) {
                DenseLayer layer = DenseLayer(layers[layerIndex]);
                currentGradient = layer.backward(currentGradient);
            } else if (_compareStrings(layerTypes[layers[layerIndex]], "Activation")) {
                ActivationLayer layer = ActivationLayer(layers[layerIndex]);
                currentGradient = layer.backward(currentGradient);
            }
        }
        
        emit BackwardPropagationCompleted(_calculateGradientMagnitude(currentGradient));
    }
    
    /**
     * @dev Update parameters semua layers
     */
    function _updateAllParameters() internal {
        for (uint256 i = 0; i < layers.length; i++) {
            if (_compareStrings(layerTypes[layers[i]], "Dense")) {
                DenseLayer layer = DenseLayer(layers[i]);
                layer.updateParameters(learningRate, address(optimizer));
                
                // Emit layer metrics
                (,,, uint256 paramCount, ) = layer.getLayerInfo();
                (uint256[][] memory weightGrads, uint256[] memory biasGrads) = layer.getGradients();
                uint256 gradMagnitude = _calculateGradientMagnitudeFromLayer(weightGrads, biasGrads);
                
                emit LayerMetrics(i, paramCount, gradMagnitude);
            }
        }
    }
    
    /**
     * @dev Reset gradients semua layers
     */
    function _resetAllGradients() internal {
        for (uint256 i = 0; i < layers.length; i++) {
            if (_compareStrings(layerTypes[layers[i]], "Dense")) {
                DenseLayer(layers[i]).resetGradients();
            } else if (_compareStrings(layerTypes[layers[i]], "Activation")) {
                ActivationLayer(layers[i]).resetState();
            }
        }
    }
    
    /**
     * @dev Mendapatkan output size dari layer
     * @param layerAddress Alamat layer
     * @return outputSize Ukuran output layer
     */
    function _getLayerOutputSize(address layerAddress) internal view returns (uint256 outputSize) {
        if (_compareStrings(layerTypes[layerAddress], "Dense")) {
            DenseLayer layer = DenseLayer(layerAddress);
            (,, uint256 _outputSize,,) = layer.getLayerInfo();
            return _outputSize;
        } else if (_compareStrings(layerTypes[layerAddress], "Activation")) {
            ActivationLayer layer = ActivationLayer(layerAddress);
            (,, uint256 _outputSize,,) = layer.getLayerInfo();
            return _outputSize;
        }
        return 0;
    }
    
    /**
     * @dev Mengecek apakah prediksi classification benar
     * @param prediction Nilai prediksi
     * @param target Nilai target
     * @return isCorrect True jika prediksi benar
     */
    function _isClassificationCorrect(uint256 prediction, uint256 target) internal pure returns (bool isCorrect) {
        // Untuk binary classification, threshold di 0.5
        uint256 threshold = PRECISION / 2;
        bool predictedClass = prediction > threshold;
        bool targetClass = target > threshold;
        
        return predictedClass == targetClass;
    }
    
    /**
     * @dev Menghitung magnitude gradien
     * @param gradient Array gradien
     * @return magnitude Magnitude gradien
     */
    function _calculateGradientMagnitude(uint256[] memory gradient) internal pure returns (uint256 magnitude) {
        uint256 sumSquares = 0;
        for (uint256 i = 0; i < gradient.length; i++) {
            sumSquares += (gradient[i] * gradient[i]) / PRECISION;
        }
        return _sqrt(sumSquares);
    }
    
    /**
     * @dev Menghitung magnitude gradien dari layer
     * @param weightGradients Gradien weights
     * @param biasGradients Gradien biases
     * @return magnitude Magnitude total gradien
     */
    function _calculateGradientMagnitudeFromLayer(
        uint256[][] memory weightGradients,
        uint256[] memory biasGradients
    ) internal pure returns (uint256 magnitude) {
        uint256 sumSquares = 0;
        
        // Weight gradients
        for (uint256 i = 0; i < weightGradients.length; i++) {
            for (uint256 j = 0; j < weightGradients[i].length; j++) {
                sumSquares += (weightGradients[i][j] * weightGradients[i][j]) / PRECISION;
            }
        }
        
        // Bias gradients
        for (uint256 i = 0; i < biasGradients.length; i++) {
            sumSquares += (biasGradients[i] * biasGradients[i]) / PRECISION;
        }
        
        return _sqrt(sumSquares);
    }
    
    /**
     * @dev Konversi array int256 ke uint256
     * @param input Array int256
     * @return output Array uint256
     */
    function _toUintArray(int256[] memory input) internal pure returns (uint256[] memory output) {
        output = new uint256[](input.length);
        for (uint256 i = 0; i < input.length; i++) {
            if (input[i] < 0) {
                output[i] = 0;
            } else {
                output[i] = uint256(input[i]);
            }
        }
        return output;
    }
    
    /**
     * @dev Membandingkan dua string
     * @param a String pertama
     * @param b String kedua
     * @return isEqual True jika string sama
     */
    function _compareStrings(string memory a, string memory b) internal pure returns (bool isEqual) {
        return keccak256(abi.encodePacked(a)) == keccak256(abi.encodePacked(b));
    }
    
    /**
     * @dev Konversi uint ke string
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
     * @dev Fungsi internal untuk menghitung square root (approximation)
     * @param x Nilai input
     * @return result Akar kuadrat dari x
     */
    function _sqrt(uint256 x) internal pure returns (uint256 result) {
        if (x == 0) return 0;
        
        // Babylonian method untuk approximation
        uint256 z = (x + 1) / 2;
        uint256 y = x;
        
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
        
        return y;
    }
}