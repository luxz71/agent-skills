// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../interfaces/IActivation.sol";

/**
 * @title Softmax
 * @dev Implementasi Softmax activation function untuk multi-class classification
 * @notice Fungsi aktivasi yang mengubah nilai menjadi probability distribution
 * @author Rizky Reza
 */
contract Softmax is IActivation {
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    uint256 private constant MAX_EXP_INPUT = 40 * 1e18; // Untuk mencegah overflow eksponensial
    
    /**
     * @dev Event khusus untuk Softmax
     * @param inputSize Ukuran input array
     * @param outputSize Ukuran output array
     */
    event SoftmaxProcessed(uint256 inputSize, uint256 outputSize);
    
    /**
     * @dev Menerapkan fungsi Softmax pada array input
     * @param input Nilai input (harus array)
     * @return output Probability distribution
     */
    function activate(uint256 input) external override pure returns (uint256 output) {
        // Softmax memerlukan array input, untuk single input gunakan array dengan satu elemen
        uint256[] memory inputs = new uint256[](1);
        inputs[0] = input;
        uint256[] memory outputs = _softmax(inputs);
        return outputs[0];
    }
    
    /**
     * @dev Menerapkan fungsi Softmax pada batch input (setiap baris adalah sample)
     * @param inputs Array nilai input (flattened array)
     * @return outputs Array nilai output setelah aktivasi
     */
    function activateBatch(
        uint256[] calldata inputs
    ) external override pure returns (uint256[] memory outputs) {
        require(inputs.length > 0, "Softmax: Input array cannot be empty");
        
        // Untuk batch processing, asumsikan inputs adalah flattened array dari multiple samples
        // Dalam implementasi nyata, perlu informasi tentang dimensi batch
        outputs = _softmax(inputs);
        
        return outputs;
    }
    
    /**
     * @dev Menghitung turunan Softmax (Jacobian matrix)
     * @param input Nilai input
     * @return derivative Diagonal element dari Jacobian matrix
     */
    function derivative(uint256 input) external override pure returns (uint256 derivative) {
        // Untuk single input, derivative adalah p_i * (1 - p_i)
        uint256[] memory inputs = new uint256[](1);
        inputs[0] = input;
        uint256[] memory outputs = _softmax(inputs);
        uint256 p = outputs[0];
        derivative = (p * (ONE - p)) / PRECISION;
        return derivative;
    }
    
    /**
     * @dev Menghitung turunan untuk batch input
     * @param inputs Array nilai input
     * @return derivatives Array nilai turunan untuk setiap input
     */
    function derivativeBatch(
        uint256[] calldata inputs
    ) external override pure returns (uint256[] memory derivatives) {
        require(inputs.length > 0, "Softmax: Input array cannot be empty");
        
        derivatives = new uint256[](inputs.length);
        uint256[] memory probabilities = _softmax(inputs);
        
        // Untuk Softmax, Jacobian matrix adalah p_i * (δ_ij - p_j)
        // Untuk diagonal elements: p_i * (1 - p_i)
        for (uint256 i = 0; i < inputs.length; i++) {
            derivatives[i] = (probabilities[i] * (ONE - probabilities[i])) / PRECISION;
        }
        
        return derivatives;
    }
    
    /**
     * @dev Menghitung turunan berdasarkan output Softmax
     * @param output Nilai output dari fungsi Softmax
     * @return derivative output * (1 - output) untuk diagonal element
     */
    function derivativeFromOutput(uint256 output) external override pure returns (uint256 derivative) {
        // Untuk diagonal element: p_i * (1 - p_i)
        derivative = (output * (ONE - output)) / PRECISION;
        return derivative;
    }
    
    /**
     * @dev Memvalidasi input untuk fungsi Softmax
     * @param input Nilai input
     * @return isValid True jika input valid
     * @return errorMessage Pesan error jika input tidak valid
     */
    function validateInput(uint256 input) external override pure returns (bool isValid, string memory errorMessage) {
        // Softmax dapat menerima semua nilai input
        return (true, "");
    }
    
    /**
     * @dev Memvalidasi batch input untuk fungsi Softmax
     * @param inputs Array nilai input
     * @return isValid True jika semua input valid
     * @return errorMessage Pesan error jika ada input tidak valid
     */
    function validateBatchInput(
        uint256[] calldata inputs
    ) external override pure returns (bool isValid, string memory errorMessage) {
        if (inputs.length == 0) {
            return (false, "Softmax: Input array cannot be empty");
        }
        return (true, "");
    }
    
    /**
     * @dev Mengembalikan informasi tentang fungsi Softmax
     * @return name "Softmax"
     * @return version "1.0.0"
     * @return activationType "Softmax"
     * @return isDifferentiable true
     * @return rangeMin 0
     * @return rangeMax 1 * PRECISION
     */
    function getActivationInfo() external override pure returns (
        string memory name,
        string memory version,
        string memory activationType,
        bool isDifferentiable,
        uint256 rangeMin,
        uint256 rangeMax
    ) {
        return (
            "Softmax",
            "1.0.0",
            "Softmax",
            true,
            0,
            ONE
        );
    }
    
    /**
     * @dev Mengembalikan properti matematika fungsi Softmax
     * @return isMonotonic false (tidak monotonik untuk individual elements)
     * @return isBounded true (output terbatas antara 0 dan 1)
     * @return hasSaturation false
     * @return zeroCentered false
     */
    function getMathematicalProperties() external override pure returns (
        bool isMonotonic,
        bool isBounded,
        bool hasSaturation,
        bool zeroCentered
    ) {
        return (false, true, false, false);
    }
    
    /**
     * @dev Mengembalikan statistik penggunaan
     * @return totalActivations Selalu 0 karena fungsi pure
     * @return averageInput Selalu 0 karena fungsi pure
     * @return averageOutput Selalu 0 karena fungsi pure
     */
    function getUsageStats() external override pure returns (
        uint256 totalActivations,
        uint256 averageInput,
        uint256 averageOutput
    ) {
        return (0, 0, 0);
    }
    
    /**
     * @dev Normalisasi input untuk Softmax (numerical stability trick)
     * @param input Nilai input
     * @return normalizedInput Nilai input yang dinormalisasi
     */
    function normalizeInput(uint256 input) external override pure returns (uint256 normalizedInput) {
        // Untuk Softmax, normalisasi dilakukan dalam fungsi utama dengan subtract max trick
        return input;
    }
    
    /**
     * @dev Denormalisasi output untuk Softmax (tidak diperlukan)
     * @param output Nilai output
     * @return denormalizedOutput Nilai output yang sama
     */
    function denormalizeOutput(uint256 output) external override pure returns (uint256 denormalizedOutput) {
        return output;
    }
    
    /**
     * @dev Mengembalikan rekomendasi range input untuk Softmax
     * @return recommendedMin 0
     * @return recommendedMax type(uint256).max
     */
    function getRecommendedInputRange() external override pure returns (
        uint256 recommendedMin,
        uint256 recommendedMax
    ) {
        return (0, type(uint256).max);
    }
    
    /**
     * @dev Mengembalikan informasi tentang stabilitas numerik
     * @return isNumericallyStable true dengan subtract max trick
     * @return overflowThreshold MAX_EXP_INPUT
     * @return underflowThreshold 0
     */
    function getNumericalStabilityInfo() external override pure returns (
        bool isNumericallyStable,
        uint256 overflowThreshold,
        uint256 underflowThreshold
    ) {
        return (true, MAX_EXP_INPUT, 0);
    }
    
    /**
     * @dev Fungsi internal untuk menghitung Softmax dengan numerical stability
     * @param inputs Array nilai input
     * @return probabilities Probability distribution
     */
    function _softmax(uint256[] memory inputs) internal pure returns (uint256[] memory probabilities) {
        uint256 length = inputs.length;
        probabilities = new uint256[](length);
        
        // Numerical stability: subtract max value untuk mencegah overflow
        uint256 maxVal = _findMax(inputs);
        
        // Hitung exponentials dan sum
        uint256 sumExp = 0;
        uint256[] memory exps = new uint256[](length);
        
        for (uint256 i = 0; i < length; i++) {
            int256 shiftedInput = int256(inputs[i]) - int256(maxVal);
            exps[i] = _exp(shiftedInput);
            sumExp += exps[i];
        }
        
        // Normalisasi untuk mendapatkan probabilities
        for (uint256 i = 0; i < length; i++) {
            if (sumExp > 0) {
                probabilities[i] = (exps[i] * PRECISION) / sumExp;
            } else {
                probabilities[i] = ONE / length; // Distribusi uniform jika sumExp == 0
            }
        }
        
        return probabilities;
    }
    
    /**
     * @dev Fungsi internal untuk mencari nilai maksimum dalam array
     * @param array Array input
     * @return maxValue Nilai maksimum
     */
    function _findMax(uint256[] memory array) internal pure returns (uint256 maxValue) {
        require(array.length > 0, "Softmax: Array cannot be empty");
        
        maxValue = array[0];
        for (uint256 i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
            }
        }
        return maxValue;
    }
    
    /**
     * @dev Fungsi internal untuk menghitung eksponensial dengan fixed-point
     * @param x Nilai eksponen dalam fixed-point
     * @return result e^x dalam fixed-point
     */
    function _exp(int256 x) internal pure returns (uint256 result) {
        // Approximation sederhana untuk e^x menggunakan Taylor series
        // Untuk implementasi produksi, gunakan library yang lebih akurat
        
        // Untuk nilai negatif besar, return 0
        if (x < -40 * int256(PRECISION)) {
            return 0;
        }
        
        // Untuk nilai positif besar, return nilai maksimum
        if (x > 40 * int256(PRECISION)) {
            return type(uint256).max / 10; // Hindari overflow
        }
        
        // Simple approximation: e^x ≈ 1 + x + x^2/2 + x^3/6
        int256 xScaled = x / int256(PRECISION);
        int256 term1 = int256(PRECISION);
        int256 term2 = xScaled;
        int256 term3 = (xScaled * xScaled) / 2;
        int256 term4 = (xScaled * xScaled * xScaled) / 6;
        
        int256 sum = term1 + term2 + term3 + term4;
        
        if (sum < 0) {
            return 0;
        }
        
        return uint256(sum) * PRECISION;
    }
}