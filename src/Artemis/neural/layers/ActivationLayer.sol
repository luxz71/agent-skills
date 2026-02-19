// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../../interfaces/IActivation.sol";

/**
 * @title ActivationLayer
 * @dev Implementasi activation layer untuk neural networks
 * @notice Layer ini menerapkan fungsi aktivasi non-linear pada input
 * @author Rizky Reza
 */
contract ActivationLayer {
    uint256 private constant PRECISION = 1e18;
    
    // Layer configuration
    uint256 public inputSize;
    IActivation public activationFunction;
    
    // Training state
    uint256[] public lastInput;
    uint256[] public lastOutput;
    
    // Events
    event LayerInitialized(uint256 inputSize, address activationFunction);
    event ForwardPassCompleted(uint256[] input, uint256[] output);
    event BackwardPassCompleted(uint256[] inputGradient, uint256[] outputGradient);
    
    /**
     * @dev Inisialisasi activation layer dengan konfigurasi
     * @param _inputSize Ukuran input layer
     * @param _activationFunction Alamat contract activation function
     */
    constructor(
        uint256 _inputSize,
        IActivation _activationFunction
    ) {
        require(_inputSize > 0, "ActivationLayer: Input size must be positive");
        require(address(_activationFunction) != address(0), "ActivationLayer: Activation function address cannot be zero");
        
        inputSize = _inputSize;
        activationFunction = _activationFunction;
        
        emit LayerInitialized(_inputSize, address(_activationFunction));
    }
    
    /**
     * @dev Melakukan forward propagation melalui activation layer
     * @param input Array input values
     * @return output Array output values setelah aktivasi
     */
    function forward(uint256[] calldata input) external returns (uint256[] memory output) {
        require(input.length == inputSize, "ActivationLayer: Input size mismatch");
        
        // Simpan input untuk backward pass
        lastInput = input;
        
        // Apply activation function
        output = activationFunction.activateBatch(input);
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
        require(outputGradient.length == inputSize, "ActivationLayer: Output gradient size mismatch");
        require(lastInput.length == inputSize, "ActivationLayer: No forward pass recorded");
        
        // Hitung turunan activation function berdasarkan input terakhir
        uint256[] memory activationDerivatives = activationFunction.derivativeBatch(lastInput);
        
        // Hitung gradien input: input_gradient = output_gradient × activation_derivative
        inputGradient = new uint256[](inputSize);
        for (uint256 i = 0; i < inputSize; i++) {
            inputGradient[i] = (outputGradient[i] * activationDerivatives[i]) / PRECISION;
        }
        
        emit BackwardPassCompleted(inputGradient, outputGradient);
        return inputGradient;
    }
    
    /**
     * @dev Melakukan backward propagation menggunakan output (lebih efisien untuk beberapa fungsi)
     * @param outputGradient Gradien dari layer berikutnya
     * @return inputGradient Gradien untuk layer sebelumnya
     */
    function backwardFromOutput(uint256[] calldata outputGradient) external returns (uint256[] memory inputGradient) {
        require(outputGradient.length == inputSize, "ActivationLayer: Output gradient size mismatch");
        require(lastOutput.length == inputSize, "ActivationLayer: No forward pass recorded");
        
        // Hitung turunan activation function berdasarkan output terakhir
        uint256[] memory activationDerivatives = new uint256[](inputSize);
        for (uint256 i = 0; i < inputSize; i++) {
            activationDerivatives[i] = activationFunction.derivativeFromOutput(lastOutput[i]);
        }
        
        // Hitung gradien input: input_gradient = output_gradient × activation_derivative
        inputGradient = new uint256[](inputSize);
        for (uint256 i = 0; i < inputSize; i++) {
            inputGradient[i] = (outputGradient[i] * activationDerivatives[i]) / PRECISION;
        }
        
        emit BackwardPassCompleted(inputGradient, outputGradient);
        return inputGradient;
    }
    
    /**
     * @dev Update parameters layer (tidak ada parameter yang perlu diupdate di activation layer)
     * @param learningRate Learning rate (tidak digunakan)
     * @param optimizer Alamat optimizer (tidak digunakan)
     */
    function updateParameters(uint256 learningRate, address optimizer) external {
        // Activation layer tidak memiliki parameter yang perlu diupdate
        // Fungsi ini tetap ada untuk konsistensi interface
    }
    
    /**
     * @dev Mengatur activation function layer
     * @param newActivationFunction Alamat contract activation function baru
     */
    function setActivationFunction(IActivation newActivationFunction) external {
        require(address(newActivationFunction) != address(0), "ActivationLayer: Activation function address cannot be zero");
        activationFunction = newActivationFunction;
    }
    
    /**
     * @dev Reset state untuk training baru
     */
    function resetState() external {
        delete lastInput;
        delete lastOutput;
    }
    
    /**
     * @dev Mengembalikan informasi tentang layer
     * @return layerType "Activation"
     * @return _inputSize Ukuran input
     * @return _outputSize Ukuran output (sama dengan input)
     * @return parameterCount Total jumlah parameter (0 untuk activation layer)
     * @return activationInfo Informasi activation function
     */
    function getLayerInfo() external view returns (
        string memory layerType,
        uint256 _inputSize,
        uint256 _outputSize,
        uint256 parameterCount,
        string memory activationInfo
    ) {
        (string memory name, , string memory activationType, , , ) = activationFunction.getActivationInfo();
        
        return (
            "Activation",
            inputSize,
            inputSize, // Output size sama dengan input size
            0, // Tidak ada parameter
            string(abi.encodePacked(name, " (", activationType, ")"))
        );
    }
    
    /**
     * @dev Mengembalikan informasi detail activation function
     * @return name Nama activation function
     * @return version Versi activation function
     * @return activationType Tipe activation
     * @return isDifferentiable Status diferensiabilitas
     * @return rangeMin Nilai minimum output
     * @return rangeMax Nilai maksimum output
     */
    function getActivationFunctionInfo() external view returns (
        string memory name,
        string memory version,
        string memory activationType,
        bool isDifferentiable,
        uint256 rangeMin,
        uint256 rangeMax
    ) {
        return activationFunction.getActivationInfo();
    }
    
    /**
     * @dev Mengembalikan properti matematika activation function
     * @return isMonotonic Status monotonisitas
     * @return isBounded Status boundedness
     * @return hasSaturation Status saturasi
     * @return zeroCentered Status zero-centered
     */
    function getActivationProperties() external view returns (
        bool isMonotonic,
        bool isBounded,
        bool hasSaturation,
        bool zeroCentered
    ) {
        return activationFunction.getMathematicalProperties();
    }
    
    /**
     * @dev Mengembalikan parameter layer (activation layer tidak memiliki parameter)
     * @return weights Array kosong (tidak ada weights)
     * @return biases Array kosong (tidak ada biases)
     */
    function getParameters() external pure returns (
        uint256[][] memory weights,
        uint256[] memory biases
    ) {
        weights = new uint256[][](0);
        biases = new uint256[](0);
        return (weights, biases);
    }
    
    /**
     * @dev Mengembalikan gradients (activation layer tidak menghitung gradients parameter)
     * @return weightGradients Array kosong
     * @return biasGradients Array kosong
     */
    function getGradients() external pure returns (
        uint256[][] memory weightGradients,
        uint256[] memory biasGradients
    ) {
        weightGradients = new uint256[][](0);
        biasGradients = new uint256[](0);
        return (weightGradients, biasGradients);
    }
    
    /**
     * @dev Validasi input untuk activation function
     * @param input Array input values
     * @return isValid True jika input valid
     * @return errorMessage Pesan error jika input tidak valid
     */
    function validateInput(uint256[] calldata input) external view returns (bool isValid, string memory errorMessage) {
        if (input.length != inputSize) {
            return (false, "ActivationLayer: Input size mismatch");
        }
        
        // Validasi setiap input menggunakan activation function
        for (uint256 i = 0; i < input.length; i++) {
            (bool valid, string memory msg) = activationFunction.validateInput(input[i]);
            if (!valid) {
                return (false, msg);
            }
        }
        
        return (true, "");
    }
    
    /**
     * @dev Normalisasi input menggunakan activation function
     * @param input Array input values
     * @return normalizedInput Array input yang sudah dinormalisasi
     */
    function normalizeInput(uint256[] calldata input) external view returns (uint256[] memory normalizedInput) {
        normalizedInput = new uint256[](input.length);
        for (uint256 i = 0; i < input.length; i++) {
            normalizedInput[i] = activationFunction.normalizeInput(input[i]);
        }
        return normalizedInput;
    }
    
    /**
     * @dev Denormalisasi output menggunakan activation function
     * @param output Array output values
     * @return denormalizedOutput Array output yang sudah didenormalisasi
     */
    function denormalizeOutput(uint256[] calldata output) external view returns (uint256[] memory denormalizedOutput) {
        denormalizedOutput = new uint256[](output.length);
        for (uint256 i = 0; i < output.length; i++) {
            denormalizedOutput[i] = activationFunction.denormalizeOutput(output[i]);
        }
        return denormalizedOutput;
    }
    
    /**
     * @dev Mengembalikan rekomendasi range input untuk activation function
     * @return recommendedMin Nilai minimum yang direkomendasikan
     * @return recommendedMax Nilai maksimum yang direkomendasikan
     */
    function getRecommendedInputRange() external view returns (
        uint256 recommendedMin,
        uint256 recommendedMax
    ) {
        return activationFunction.getRecommendedInputRange();
    }
    
    /**
     * @dev Mengembalikan informasi stabilitas numerik activation function
     * @return isNumericallyStable Status stabilitas numerik
     * @return overflowThreshold Threshold untuk overflow
     * @return underflowThreshold Threshold untuk underflow
     */
    function getNumericalStabilityInfo() external view returns (
        bool isNumericallyStable,
        uint256 overflowThreshold,
        uint256 underflowThreshold
    ) {
        return activationFunction.getNumericalStabilityInfo();
    }
}