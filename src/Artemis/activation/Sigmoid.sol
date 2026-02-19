// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../interfaces/IActivation.sol";

/**
 * @title Sigmoid
 * @dev Implementasi Sigmoid activation function
 * @notice Fungsi aktivasi non-linear yang mengembalikan 1 / (1 + e^(-x))
 * @author Rizky Reza
 */
contract Sigmoid is IActivation {
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    int256 private constant MAX_STABLE_VALUE = 20 * 1e18; // Untuk mencegah overflow
    int256 private constant MIN_STABLE_VALUE = -20 * 1e18; // Untuk mencegah underflow
    
    /**
     * @dev Event khusus untuk Sigmoid
     * @param input Nilai input sebelum aktivasi
     * @param output Nilai output setelah aktivasi
     */
    event SigmoidActivated(uint256 input, uint256 output);
    
    /**
     * @dev Menerapkan fungsi Sigmoid pada input dengan numerical stability
     * @param input Nilai input
     * @return output 1 / (1 + e^(-x))
     */
    function activate(uint256 input) external override pure returns (uint256 output) {
        // Untuk numerical stability, gunakan approximation untuk nilai ekstrem
        int256 signedInput = int256(input);
        if (signedInput >= MAX_STABLE_VALUE) {
            return ONE; // Mendekati 1.0
        } else if (signedInput <= MIN_STABLE_VALUE) {
            return 0; // Mendekati 0
        }
        
        // Hitung e^(-x) menggunakan approximation untuk fixed-point
        uint256 expNegX = _exp(-signedInput);
        
        // Sigmoid: 1 / (1 + e^(-x))
        output = (ONE * PRECISION) / (ONE + expNegX);
        
        return output;
    }
    
    /**
     * @dev Menerapkan fungsi Sigmoid pada batch input
     * @param inputs Array nilai input
     * @return outputs Array nilai output setelah aktivasi
     */
    function activateBatch(
        uint256[] calldata inputs
    ) external override pure returns (uint256[] memory outputs) {
        require(inputs.length > 0, "Sigmoid: Input array cannot be empty");
        
        outputs = new uint256[](inputs.length);
        
        for (uint256 i = 0; i < inputs.length; i++) {
            outputs[i] = _activate(inputs[i]);
        }
        
        return outputs;
    }
    
    /**
     * @dev Menghitung turunan Sigmoid pada input
     * @param input Nilai input
     * @return derivative sigmoid(x) * (1 - sigmoid(x))
     */
    function derivative(uint256 input) external override pure returns (uint256 derivative) {
        uint256 sigmoidOutput = _activate(input);
        
        // Derivative: sigmoid(x) * (1 - sigmoid(x))
        derivative = (sigmoidOutput * (ONE - sigmoidOutput)) / PRECISION;
        
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
        require(inputs.length > 0, "Sigmoid: Input array cannot be empty");
        
        derivatives = new uint256[](inputs.length);
        
        for (uint256 i = 0; i < inputs.length; i++) {
            derivatives[i] = _derivative(inputs[i]);
        }
        
        return derivatives;
    }
    
    /**
     * @dev Menghitung turunan berdasarkan output Sigmoid
     * @param output Nilai output dari fungsi Sigmoid
     * @return derivative output * (1 - output)
     */
    function derivativeFromOutput(uint256 output) external override pure returns (uint256 derivative) {
        // Derivative berdasarkan output: output * (1 - output)
        derivative = (output * (ONE - output)) / PRECISION;
        
        return derivative;
    }
    
    /**
     * @dev Memvalidasi input untuk fungsi Sigmoid
     * @param input Nilai input
     * @return isValid True jika input valid
     * @return errorMessage Pesan error jika input tidak valid
     */
    function validateInput(uint256 input) external override pure returns (bool isValid, string memory errorMessage) {
        // Sigmoid dapat menerima semua nilai input, tapi peringatkan untuk nilai ekstrem
        int256 signedInput = int256(input);
        if (signedInput >= MAX_STABLE_VALUE || signedInput <= MIN_STABLE_VALUE) {
            return (true, "Warning: Input value may cause numerical instability");
        }
        return (true, "");
    }
    
    /**
     * @dev Memvalidasi batch input untuk fungsi Sigmoid
     * @param inputs Array nilai input
     * @return isValid True jika semua input valid
     * @return errorMessage Pesan error jika ada input tidak valid
     */
    function validateBatchInput(
        uint256[] calldata inputs
    ) external override pure returns (bool isValid, string memory errorMessage) {
        if (inputs.length == 0) {
            return (false, "Sigmoid: Input array cannot be empty");
        }
        
        for (uint256 i = 0; i < inputs.length; i++) {
            int256 signedInput = int256(inputs[i]);
            if (signedInput >= MAX_STABLE_VALUE || signedInput <= MIN_STABLE_VALUE) {
                return (true, "Warning: Some input values may cause numerical instability");
            }
        }
        
        return (true, "");
    }
    
    /**
     * @dev Mengembalikan informasi tentang fungsi Sigmoid
     * @return name "Sigmoid"
     * @return version "1.0.0"
     * @return activationType "Sigmoid"
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
            "Sigmoid",
            "1.0.0",
            "Sigmoid",
            true,
            0,
            ONE
        );
    }
    
    /**
     * @dev Mengembalikan properti matematika fungsi Sigmoid
     * @return isMonotonic true (fungsi monotonik naik)
     * @return isBounded true (output terbatas antara 0 dan 1)
     * @return hasSaturation true (memiliki daerah saturasi di ekstrem)
     * @return zeroCentered false (output tidak terpusat di nol)
     */
    function getMathematicalProperties() external override pure returns (
        bool isMonotonic,
        bool isBounded,
        bool hasSaturation,
        bool zeroCentered
    ) {
        return (true, true, true, false);
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
     * @dev Normalisasi input untuk Sigmoid (scaling ke range yang sesuai)
     * @param input Nilai input
     * @return normalizedInput Nilai input yang dinormalisasi
     */
    function normalizeInput(uint256 input) external override pure returns (uint256 normalizedInput) {
        // Untuk Sigmoid, normalisasi ke range [-10, 10] untuk stabilitas
        int256 signedInput = int256(input);
        if (signedInput > 10 * int256(PRECISION)) {
            return 10 * PRECISION;
        } else if (signedInput < -10 * int256(PRECISION)) {
            return 0; // Tidak bisa return negatif, jadi return 0
        }
        return input;
    }
    
    /**
     * @dev Denormalisasi output untuk Sigmoid (tidak diperlukan khusus)
     * @param output Nilai output
     * @return denormalizedOutput Nilai output yang sama
     */
    function denormalizeOutput(uint256 output) external override pure returns (uint256 denormalizedOutput) {
        return output;
    }
    
    /**
     * @dev Mengembalikan rekomendasi range input untuk Sigmoid
     * @return recommendedMin -10 * PRECISION
     * @return recommendedMax 10 * PRECISION
     */
    function getRecommendedInputRange() external override pure returns (
        uint256 recommendedMin,
        uint256 recommendedMax
    ) {
        return (0, 10 * PRECISION); // Hanya bisa return nilai positif
    }
    
    /**
     * @dev Mengembalikan informasi tentang stabilitas numerik
     * @return isNumericallyStable true dengan precautions
     * @return overflowThreshold MAX_STABLE_VALUE
     * @return underflowThreshold MIN_STABLE_VALUE
     */
    function getNumericalStabilityInfo() external override pure returns (
        bool isNumericallyStable,
        uint256 overflowThreshold,
        uint256 underflowThreshold
    ) {
        return (true, uint256(MAX_STABLE_VALUE), 0); // Underflow threshold tidak relevan untuk uint
    }
    
    /**
     * @dev Fungsi internal untuk menghitung eksponensial dengan fixed-point
     * @param x Nilai eksponen dalam fixed-point
     * @return result e^x dalam fixed-point
     */
    /**
     * @dev Fungsi internal untuk aktivasi Sigmoid
     * @param input Nilai input
     * @return output Hasil aktivasi
     */
    function _activate(uint256 input) internal pure returns (uint256 output) {
        // Untuk numerical stability, gunakan approximation untuk nilai ekstrem
        int256 signedInput = int256(input);
        if (signedInput >= MAX_STABLE_VALUE) {
            return ONE; // Mendekati 1.0
        } else if (signedInput <= MIN_STABLE_VALUE) {
            return 0; // Mendekati 0
        }
        
        // Hitung e^(-x) menggunakan approximation untuk fixed-point
        uint256 expNegX = _exp(-signedInput);
        
        // Sigmoid: 1 / (1 + e^(-x))
        output = (ONE * PRECISION) / (ONE + expNegX);
        
        return output;
    }
    
    /**
     * @dev Fungsi internal untuk menghitung turunan Sigmoid
     * @param input Nilai input
     * @return derivative Turunan Sigmoid
     */
    function _derivative(uint256 input) internal pure returns (uint256 derivative) {
        uint256 sigmoidOutput = _activate(input);
        
        // Derivative: sigmoid(x) * (1 - sigmoid(x))
        derivative = (sigmoidOutput * (ONE - sigmoidOutput)) / PRECISION;
        
        return derivative;
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