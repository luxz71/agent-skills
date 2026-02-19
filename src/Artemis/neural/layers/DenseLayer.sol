// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../../interfaces/IActivation.sol";
import "../../math/Matrix.sol";
import "../../math/ArrayUtils.sol";

/**
 * @title DenseLayer
 * @dev Implementasi fully connected layer untuk neural networks
 * @notice Layer ini melakukan linear transformation: output = input × weights + bias
 * @author Rizky Reza
 */
contract DenseLayer {
    using Matrix for uint256[][];
    
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    
    // Layer configuration
    uint256 public inputSize;
    uint256 public outputSize;
    bool public useBias;
    IActivation public activationFunction;
    
    // Parameters
    uint256[][] public weights;
    uint256[] public biases;
    
    // Training state
    uint256[][] public weightGradients;
    uint256[] public biasGradients;
    uint256[] public lastInput;
    uint256[] public lastPreActivation;
    uint256[] public lastOutput;
    
    // Events
    event LayerInitialized(uint256 inputSize, uint256 outputSize, bool useBias);
    event ForwardPassCompleted(uint256[] input, uint256[] output);
    event BackwardPassCompleted(uint256[] inputGradient, uint256[][] weightGradient, uint256[] biasGradient);
    event ParametersUpdated(uint256[][] newWeights, uint256[] newBiases);
    
    /**
     * @dev Inisialisasi dense layer dengan konfigurasi
     * @param _inputSize Ukuran input layer
     * @param _outputSize Ukuran output layer
     * @param _useBias True jika menggunakan bias
     * @param _activationFunction Alamat contract activation function
     */
    constructor(
        uint256 _inputSize,
        uint256 _outputSize,
        bool _useBias,
        IActivation _activationFunction
    ) {
        require(_inputSize > 0, "DenseLayer: Input size must be positive");
        require(_outputSize > 0, "DenseLayer: Output size must be positive");
        
        inputSize = _inputSize;
        outputSize = _outputSize;
        useBias = _useBias;
        activationFunction = _activationFunction;
        
        // Initialize weights dengan Xavier/Glorot initialization
        _initializeWeights();
        
        // Initialize biases dengan zeros
        if (useBias) {
            _initializeBiases();
        }
        
        emit LayerInitialized(_inputSize, _outputSize, _useBias);
    }
    
    /**
     * @dev Melakukan forward propagation melalui layer
     * @param input Array input values
     * @return output Array output values setelah aktivasi
     */
    function forward(uint256[] calldata input) external returns (uint256[] memory output) {
        require(input.length == inputSize, "DenseLayer: Input size mismatch");
        
        // Simpan input untuk backward pass
        lastInput = input;
        
        // Linear transformation: output = input × weights + bias
        uint256[] memory preActivation = new uint256[](outputSize);
        
        for (uint256 i = 0; i < outputSize; i++) {
            uint256 sum = 0;
            for (uint256 j = 0; j < inputSize; j++) {
                sum += (input[j] * weights[i][j]) / PRECISION;
            }
            
            if (useBias) {
                sum += biases[i];
            }
            
            preActivation[i] = sum;
        }
        
        // Simpan pre-activation untuk backward pass
        lastPreActivation = preActivation;
        
        // Apply activation function
        output = activationFunction.activateBatch(preActivation);
        lastOutput = output;
        
        emit ForwardPassCompleted(input, output);
        return output;
    }
    
    /**
     * @dev Melakukan backward propagation untuk menghitung gradients
     * @param outputGradient Gradien dari layer berikutnya
     * @return inputGradient Gradien untuk layer sebelumnya
     */
    function backward(uint256[] calldata outputGradient) external returns (uint256[] memory inputGradient) {
        require(outputGradient.length == outputSize, "DenseLayer: Output gradient size mismatch");
        require(lastInput.length == inputSize, "DenseLayer: No forward pass recorded");
        
        // Hitung gradien untuk pre-activation (menggunakan derivative activation function)
        uint256[] memory activationDerivatives = activationFunction.derivativeBatch(lastPreActivation);
        uint256[] memory preActivationGradient = new uint256[](outputSize);
        
        for (uint256 i = 0; i < outputSize; i++) {
            preActivationGradient[i] = (outputGradient[i] * activationDerivatives[i]) / PRECISION;
        }
        
        // Hitung gradien untuk weights
        weightGradients = new uint256[][](outputSize);
        for (uint256 i = 0; i < outputSize; i++) {
            weightGradients[i] = new uint256[](inputSize);
            for (uint256 j = 0; j < inputSize; j++) {
                weightGradients[i][j] = (preActivationGradient[i] * lastInput[j]) / PRECISION;
            }
        }
        
        // Hitung gradien untuk biases (jika digunakan)
        if (useBias) {
            biasGradients = preActivationGradient;
        }
        
        // Hitung gradien untuk input (untuk layer sebelumnya)
        inputGradient = new uint256[](inputSize);
        for (uint256 j = 0; j < inputSize; j++) {
            uint256 sum = 0;
            for (uint256 i = 0; i < outputSize; i++) {
                sum += (preActivationGradient[i] * weights[i][j]) / PRECISION;
            }
            inputGradient[j] = sum;
        }
        
        emit BackwardPassCompleted(inputGradient, weightGradients, biasGradients);
        return inputGradient;
    }
    
    /**
     * @dev Update parameters layer menggunakan gradients yang sudah dihitung
     * @param learningRate Learning rate untuk update
     * @param optimizer Alamat optimizer (jika digunakan)
     */
    function updateParameters(uint256 learningRate, address optimizer) external {
        require(weightGradients.length > 0, "DenseLayer: No gradients calculated");
        
        // Update weights
        for (uint256 i = 0; i < outputSize; i++) {
            for (uint256 j = 0; j < inputSize; j++) {
                int256 gradient = int256(weightGradients[i][j]);
                weights[i][j] = _updateParameter(weights[i][j], gradient, learningRate);
            }
        }
        
        // Update biases (jika digunakan)
        if (useBias && biasGradients.length > 0) {
            for (uint256 i = 0; i < outputSize; i++) {
                int256 gradient = int256(biasGradients[i]);
                biases[i] = _updateParameter(biases[i], gradient, learningRate);
            }
        }
        
        emit ParametersUpdated(weights, biases);
    }
    
    /**
     * @dev Mengatur weights dan biases layer
     * @param newWeights Array 2D weights baru
     * @param newBiases Array biases baru
     */
    function setParameters(uint256[][] calldata newWeights, uint256[] calldata newBiases) external {
        require(newWeights.length == outputSize, "DenseLayer: Weights rows mismatch");
        require(newWeights[0].length == inputSize, "DenseLayer: Weights columns mismatch");
        
        if (useBias) {
            require(newBiases.length == outputSize, "DenseLayer: Biases size mismatch");
            biases = newBiases;
        }
        
        weights = newWeights;
        emit ParametersUpdated(newWeights, newBiases);
    }
    
    /**
     * @dev Reset gradients untuk training baru
     */
    function resetGradients() external {
        delete weightGradients;
        delete biasGradients;
        delete lastInput;
        delete lastPreActivation;
        delete lastOutput;
    }
    
    /**
     * @dev Mengembalikan informasi tentang layer
     * @return layerType "Dense"
     * @return _inputSize Ukuran input
     * @return _outputSize Ukuran output
     * @return parameterCount Total jumlah parameter
     * @return _useBias Status penggunaan bias
     */
    function getLayerInfo() external view returns (
        string memory layerType,
        uint256 _inputSize,
        uint256 _outputSize,
        uint256 parameterCount,
        bool _useBias
    ) {
        uint256 totalParams = outputSize * inputSize;
        if (useBias) {
            totalParams += outputSize;
        }
        
        return (
            "Dense",
            inputSize,
            outputSize,
            totalParams,
            useBias
        );
    }
    
    /**
     * @dev Mengembalikan parameter layer
     * @return _weights Array 2D weights
     * @return _biases Array biases
     */
    function getParameters() external view returns (
        uint256[][] memory _weights,
        uint256[] memory _biases
    ) {
        return (weights, biases);
    }
    
    /**
     * @dev Mengembalikan gradients yang terakhir dihitung
     * @return _weightGradients Gradien weights
     * @return _biasGradients Gradien biases
     */
    function getGradients() external view returns (
        uint256[][] memory _weightGradients,
        uint256[] memory _biasGradients
    ) {
        return (weightGradients, biasGradients);
    }
    
    /**
     * @dev Fungsi internal untuk inisialisasi weights dengan Xavier initialization
     */
    function _initializeWeights() internal {
        weights = new uint256[][](outputSize);
        
        // Hitung scaling factor untuk Xavier initialization
        uint256 scale = PRECISION / _sqrt(inputSize + outputSize);
        
        for (uint256 i = 0; i < outputSize; i++) {
            weights[i] = new uint256[](inputSize);
            for (uint256 j = 0; j < inputSize; j++) {
                // Generate random-like weight dalam range [-scale, scale]
                // Untuk simplicity, gunakan deterministic initialization
                uint256 randomValue = uint256(keccak256(abi.encodePacked(i, j, block.timestamp))) % (2 * scale);
                weights[i][j] = randomValue; // Range: 0 hingga 2*scale
            }
        }
    }
    
    /**
     * @dev Fungsi internal untuk inisialisasi biases dengan zeros
     */
    function _initializeBiases() internal {
        biases = new uint256[](outputSize);
        for (uint256 i = 0; i < outputSize; i++) {
            biases[i] = 0;
        }
    }
    
    /**
     * @dev Fungsi internal untuk update parameter individual
     * @param parameter Parameter saat ini
     * @param gradient Gradien parameter
     * @param learningRate Learning rate
     * @return updatedParameter Parameter yang sudah diupdate
     */
    function _updateParameter(
        uint256 parameter,
        int256 gradient,
        uint256 learningRate
    ) internal pure returns (uint256 updatedParameter) {
        int256 parameterInt = int256(parameter);
        int256 update = (gradient * int256(learningRate)) / int256(PRECISION);
        
        int256 newParameter = parameterInt - update;
        
        // Pastikan parameter tidak negatif
        if (newParameter < 0) {
            return 0;
        }
        
        return uint256(newParameter);
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