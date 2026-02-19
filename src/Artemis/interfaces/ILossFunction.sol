// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ILossFunction
 * @dev Interface untuk fungsi loss dalam machine learning
 * @notice Interface ini mendefinisikan kontrak untuk menghitung error antara prediksi dan target
 * @author Rizky Reza
 */
interface ILossFunction {
    /**
     * @dev Event yang dipancarkan ketika loss dihitung
     * @param lossFunctionAddress Alamat contract loss function
     * @param lossValue Nilai loss yang dihitung
     * @param batchSize Ukuran batch data
     */
    event LossCalculated(
        address indexed lossFunctionAddress,
        uint256 lossValue,
        uint256 batchSize
    );

    /**
     * @dev Menghitung loss untuk single sample
     * @param prediction Prediksi model
     * @param target Target yang sebenarnya
     * @return loss Nilai loss untuk sample ini
     */
    function calculateLoss(
        uint256 prediction,
        uint256 target
    ) external pure returns (uint256 loss);

    /**
     * @dev Menghitung loss untuk batch data
     * @param predictions Array prediksi model
     * @param targets Array target yang sebenarnya
     * @return totalLoss Total loss untuk batch
     * @return averageLoss Rata-rata loss untuk batch
     */
    function calculateBatchLoss(
        uint256[] calldata predictions,
        uint256[] calldata targets
    ) external pure returns (uint256 totalLoss, uint256 averageLoss);

    /**
     * @dev Menghitung gradien loss terhadap prediksi
     * @param prediction Prediksi model
     * @param target Target yang sebenarnya
     * @return gradient Gradien loss terhadap prediksi
     */
    function calculateGradient(
        uint256 prediction,
        uint256 target
    ) external pure returns (int256 gradient);

    /**
     * @dev Menghitung gradien untuk batch data
     * @param predictions Array prediksi model
     * @param targets Array target yang sebenarnya
     * @return gradients Array gradien untuk setiap sample
     * @return averageGradient Rata-rata gradien
     */
    function calculateBatchGradient(
        uint256[] calldata predictions,
        uint256[] calldata targets
    ) external pure returns (int256[] memory gradients, int256 averageGradient);

    /**
     * @dev Memvalidasi input untuk perhitungan loss
     * @param predictions Array prediksi model
     * @param targets Array target yang sebenarnya
     * @return isValid True jika input valid
     * @return errorMessage Pesan error jika input tidak valid
     */
    function validateInput(
        uint256[] calldata predictions,
        uint256[] calldata targets
    ) external pure returns (bool isValid, string memory errorMessage);

    /**
     * @dev Mengembalikan informasi tentang loss function
     * @return name Nama loss function
     * @return version Versi loss function
     * @return isDifferentiable True jika fungsi loss dapat didiferensiasi
     * @return rangeMin Nilai minimum yang mungkin untuk loss
     * @return rangeMax Nilai maksimum yang mungkin untuk loss
     */
    function getLossFunctionInfo() external pure returns (
        string memory name,
        string memory version,
        bool isDifferentiable,
        uint256 rangeMin,
        uint256 rangeMax
    );

    /**
     * @dev Mengembalikan konfigurasi loss function
     * @return supportsBatch True jika mendukung perhitungan batch
     * @return requiresNormalization True jika memerlukan normalisasi input
     * @return precision Precision yang digunakan (dalam fixed-point)
     */
    function getConfiguration() external pure returns (
        bool supportsBatch,
        bool requiresNormalization,
        uint256 precision
    );
}