// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../interfaces/IActivation.sol";

/**
 * @title ReLU
 * @dev Implementasi Rectified Linear Unit (ReLU) activation function
 * @notice Fungsi aktivasi non-linear yang mengembalikan max(0, x)
 * @author Rizky Reza
 */
contract ReLU is IActivation {
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ZERO_THRESHOLD = 1e12; // Threshold untuk nilai mendekati nol
    
    /**
     * @dev Event khusus untuk ReLU
     * @param input Nilai input sebelum aktivasi
     * @param output Nilai output setelah aktivasi
     */
    event ReLUActivated(uint256 input, uint256 output);
    
    /**
     * @dev Menerapkan fungsi ReLU pada input
     * @param input Nilai input
     * @return output max(0, input)
     */
    function activate(uint256 input) external override pure returns (uint256 output) {
        // ReLU: max(0, x)
        if (input > 0) {
            return input;
        } else {
            return 0;
        }
    }
    
    /**
     * @dev Menerapkan fungsi ReLU pada batch input
     * @param inputs Array nilai input
     * @return outputs Array nilai output setelah aktivasi
     */
    function activateBatch(
        uint256[] calldata inputs
    ) external override pure returns (uint256[] memory outputs) {
        require(inputs.length > 0, "ReLU: Input array cannot be empty");
        
        outputs = new uint256[](inputs.length);
        
        for (uint256 i = 0; i < inputs.length; i++) {
            if (inputs[i] > 0) {
                outputs[i] = inputs[i];
            } else {
                outputs[i] = 0;
            }
        }
        
        return outputs;
    }
    
    /**
     * @dev Menghitung turunan ReLU pada input
     * @param input Nilai input
     * @return derivative 1 jika input > 0, 0 jika input <= 0
     */
    function derivative(uint256 input) external override pure returns (uint256 derivative) {
        // Derivative ReLU: 1 jika x > 0, 0 jika x <= 0
        if (input > 0) {
            return PRECISION; // 1.0 dalam fixed-point
        } else {
            return 0;
        }
    }
    
    /**
     * @dev Menghitung turunan untuk batch input
     * @param inputs Array nilai input
     * @return derivatives Array nilai turunan untuk setiap input
     */
    function derivativeBatch(
        uint256[] calldata inputs
    ) external override pure returns (uint256[] memory derivatives) {
        require(inputs.length > 0, "ReLU: Input array cannot be empty");
        
        derivatives = new uint256[](inputs.length);
        
        for (uint256 i = 0; i < inputs.length; i++) {
            if (inputs[i] > 0) {
                derivatives[i] = PRECISION; // 1.0
            } else {
                derivatives[i] = 0;
            }
        }
        
        return derivatives;
    }
    
    /**
     * @dev Menghitung turunan berdasarkan output ReLU
     * @param output Nilai output dari fungsi ReLU
     * @return derivative 1 jika output > 0, 0 jika output == 0
     */
    function derivativeFromOutput(uint256 output) external override pure returns (uint256 derivative) {
        // Untuk ReLU, jika output > 0 maka input > 0, jadi derivative = 1
        // Jika output == 0, maka input <= 0, jadi derivative = 0
        if (output > 0) {
            return PRECISION; // 1.0
        } else {
            return 0;
        }
    }
    
    /**
     * @dev Memvalidasi input untuk fungsi ReLU
     * @param input Nilai input
     * @return isValid Selalu true untuk ReLU (tidak ada batasan input)
     * @return errorMessage String kosong
     */
    function validateInput(uint256 input) external override pure returns (bool isValid, string memory errorMessage) {
        // ReLU dapat menerima semua nilai input
        return (true, "");
    }
    
    /**
     * @dev Memvalidasi batch input untuk fungsi ReLU
     * @param inputs Array nilai input
     * @return isValid Selalu true untuk ReLU
     * @return errorMessage String kosong
     */
    function validateBatchInput(
        uint256[] calldata inputs
    ) external override pure returns (bool isValid, string memory errorMessage) {
        if (inputs.length == 0) {
            return (false, "ReLU: Input array cannot be empty");
        }
        return (true, "");
    }
    
    /**
     * @dev Mengembalikan informasi tentang fungsi ReLU
     * @return name "ReLU"
     * @return version "1.0.0"
     * @return activationType "ReLU"
     * @return isDifferentiable true (piecewise differentiable)
     * @return rangeMin 0
     * @return rangeMax type(uint256).max
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
            "ReLU",
            "1.0.0",
            "ReLU",
            true,
            0,
            type(uint256).max
        );
    }
    
    /**
     * @dev Mengembalikan properti matematika fungsi ReLU
     * @return isMonotonic true (fungsi monotonik naik)
     * @return isBounded false (output tidak terbatas di atas)
     * @return hasSaturation true (memiliki daerah saturasi di x <= 0)
     * @return zeroCentered false (output tidak terpusat di nol)
     */
    function getMathematicalProperties() external override pure returns (
        bool isMonotonic,
        bool isBounded,
        bool hasSaturation,
        bool zeroCentered
    ) {
        return (true, false, true, false);
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
     * @dev Normalisasi input untuk ReLU (tidak diperlukan normalisasi khusus)
     * @param input Nilai input
     * @return normalizedInput Nilai input yang sama
     */
    function normalizeInput(uint256 input) external override pure returns (uint256 normalizedInput) {
        return input;
    }
    
    /**
     * @dev Denormalisasi output untuk ReLU (tidak diperlukan denormalisasi khusus)
     * @param output Nilai output
     * @return denormalizedOutput Nilai output yang sama
     */
    function denormalizeOutput(uint256 output) external override pure returns (uint256 denormalizedOutput) {
        return output;
    }
    
    /**
     * @dev Mengembalikan rekomendasi range input untuk ReLU
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
     * @return isNumericallyStable true (stabil secara numerik)
     * @return overflowThreshold type(uint256).max
     * @return underflowThreshold 0
     */
    function getNumericalStabilityInfo() external override pure returns (
        bool isNumericallyStable,
        uint256 overflowThreshold,
        uint256 underflowThreshold
    ) {
        return (true, type(uint256).max, 0);
    }
}