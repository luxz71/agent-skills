// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title IActivation
 * @dev Interface untuk fungsi aktivasi dalam neural networks
 * @notice Interface ini mendefinisikan kontrak untuk fungsi aktivasi non-linear
 * @author Rizky Reza
 */
interface IActivation {
    /**
     * @dev Event yang dipancarkan ketika fungsi aktivasi diproses
     * @param activationAddress Alamat contract activation function
     * @param inputSize Ukuran input yang diproses
     * @param outputSize Ukuran output yang dihasilkan
     */
    event ActivationProcessed(
        address indexed activationAddress,
        uint256 inputSize,
        uint256 outputSize
    );

    /**
     * @dev Menerapkan fungsi aktivasi pada input
     * @param input Nilai input
     * @return output Nilai output setelah aktivasi
     */
    function activate(uint256 input) external pure returns (uint256 output);

    /**
     * @dev Menerapkan fungsi aktivasi pada array input (batch processing)
     * @param inputs Array nilai input
     * @return outputs Array nilai output setelah aktivasi
     */
    function activateBatch(
        uint256[] calldata inputs
    ) external pure returns (uint256[] memory outputs);

    /**
     * @dev Menghitung turunan fungsi aktivasi pada input
     * @param input Nilai input
     * @return derivative Nilai turunan pada input tersebut
     */
    function derivative(uint256 input) external pure returns (uint256 derivative);

    /**
     * @dev Menghitung turunan untuk array input (batch processing)
     * @param inputs Array nilai input
     * @return derivatives Array nilai turunan untuk setiap input
     */
    function derivativeBatch(
        uint256[] calldata inputs
    ) external pure returns (uint256[] memory derivatives);

    /**
     * @dev Menghitung turunan berdasarkan output (untuk backpropagation)
     * @param output Nilai output dari fungsi aktivasi
     * @return derivative Nilai turunan berdasarkan output
     */
    function derivativeFromOutput(uint256 output) external pure returns (uint256 derivative);

    /**
     * @dev Memvalidasi input untuk fungsi aktivasi
     * @param input Nilai input
     * @return isValid True jika input valid
     * @return errorMessage Pesan error jika input tidak valid
     */
    function validateInput(uint256 input) external pure returns (bool isValid, string memory errorMessage);

    /**
     * @dev Memvalidasi batch input untuk fungsi aktivasi
     * @param inputs Array nilai input
     * @return isValid True jika semua input valid
     * @return errorMessage Pesan error jika ada input tidak valid
     */
    function validateBatchInput(
        uint256[] calldata inputs
    ) external pure returns (bool isValid, string memory errorMessage);

    /**
     * @dev Mengembalikan informasi tentang fungsi aktivasi
     * @return name Nama fungsi aktivasi
     * @return version Versi fungsi aktivasi
     * @return activationType Tipe aktivasi (ReLU, Sigmoid, Tanh, dll.)
     * @return isDifferentiable True jika fungsi dapat didiferensiasi
     * @return rangeMin Nilai minimum output
     * @return rangeMax Nilai maksimum output
     */
    function getActivationInfo() external pure returns (
        string memory name,
        string memory version,
        string memory activationType,
        bool isDifferentiable,
        uint256 rangeMin,
        uint256 rangeMax
    );

    /**
     * @dev Mengembalikan properti matematika fungsi aktivasi
     * @return isMonotonic True jika fungsi monotonik
     * @return isBounded True jika output terbatas
     * @return hasSaturation True jika memiliki daerah saturasi
     * @return zeroCentered True jika output terpusat di nol
     */
    function getMathematicalProperties() external pure returns (
        bool isMonotonic,
        bool isBounded,
        bool hasSaturation,
        bool zeroCentered
    );

    /**
     * @dev Mengembalikan statistik penggunaan
     * @return totalActivations Total jumlah aktivasi yang dilakukan
     * @return averageInput Rata-rata nilai input
     * @return averageOutput Rata-rata nilai output
     */
    function getUsageStats() external view returns (
        uint256 totalActivations,
        uint256 averageInput,
        uint256 averageOutput
    );

    /**
     * @dev Normalisasi input untuk fungsi aktivasi tertentu
     * @param input Nilai input
     * @return normalizedInput Nilai input yang sudah dinormalisasi
     */
    function normalizeInput(uint256 input) external pure returns (uint256 normalizedInput);

    /**
     * @dev Denormalisasi output untuk fungsi aktivasi tertentu
     * @param output Nilai output
     * @return denormalizedOutput Nilai output yang sudah didenormalisasi
     */
    function denormalizeOutput(uint256 output) external pure returns (uint256 denormalizedOutput);

    /**
     * @dev Mengembalikan rekomendasi range input untuk fungsi aktivasi
     * @return recommendedMin Nilai minimum input yang direkomendasikan
     * @return recommendedMax Nilai maksimum input yang direkomendasikan
     */
    function getRecommendedInputRange() external pure returns (
        uint256 recommendedMin,
        uint256 recommendedMax
    );

    /**
     * @dev Mengembalikan informasi tentang stabilitas numerik
     * @return isNumericallyStable True jika stabil secara numerik
     * @return overflowThreshold Threshold untuk overflow
     * @return underflowThreshold Threshold untuk underflow
     */
    function getNumericalStabilityInfo() external pure returns (
        bool isNumericallyStable,
        uint256 overflowThreshold,
        uint256 underflowThreshold
    );
}