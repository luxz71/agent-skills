// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title IOptimizer
 * @dev Interface untuk optimizers dalam machine learning
 * @notice Interface ini mendefinisikan kontrak untuk mengoptimalkan parameter model
 * @author Rizky Reza
 */
interface IOptimizer {
    /**
     * @dev Event yang dipancarkan ketika optimizer melakukan update parameter
     * @param optimizerAddress Alamat contract optimizer
     * @param parameterCount Jumlah parameter yang diupdate
     * @param averageUpdate Rata-rata perubahan parameter
     */
    event ParametersUpdated(
        address indexed optimizerAddress,
        uint256 parameterCount,
        int256 averageUpdate
    );

    /**
     * @dev Event yang dipancarkan ketika learning rate diupdate
     * @param optimizerAddress Alamat contract optimizer
     * @param oldLearningRate Learning rate sebelumnya
     * @param newLearningRate Learning rate baru
     */
    event LearningRateUpdated(
        address indexed optimizerAddress,
        uint256 oldLearningRate,
        uint256 newLearningRate
    );

    /**
     * @dev Mengupdate parameter model berdasarkan gradien
     * @param parameters Array parameter saat ini
     * @param gradients Array gradien untuk setiap parameter
     * @param learningRate Learning rate untuk optimisasi
     * @return updatedParameters Array parameter yang sudah diupdate
     * @return updateMagnitude Besarnya perubahan parameter
     */
    function updateParameters(
        uint256[] calldata parameters,
        int256[] calldata gradients,
        uint256 learningRate
    ) external returns (
        uint256[] memory updatedParameters,
        uint256 updateMagnitude
    );

    /**
     * @dev Mengupdate parameter untuk single parameter
     * @param parameter Parameter saat ini
     * @param gradient Gradien untuk parameter
     * @param learningRate Learning rate untuk optimisasi
     * @return updatedParameter Parameter yang sudah diupdate
     */
    function updateParameter(
        uint256 parameter,
        int256 gradient,
        uint256 learningRate
    ) external returns (uint256 updatedParameter);

    /**
     * @dev Mengatur learning rate optimizer
     * @param newLearningRate Learning rate baru
     * @return success Status keberhasilan pengaturan learning rate
     */
    function setLearningRate(uint256 newLearningRate) external returns (bool success);

    /**
     * @dev Mengatur momentum untuk optimizers yang mendukung momentum
     * @param momentum Nilai momentum (0-1 dalam fixed-point)
     * @return success Status keberhasilan pengaturan momentum
     */
    function setMomentum(uint256 momentum) external returns (bool success);

    /**
     * @dev Reset state optimizer (misalnya untuk momentum, velocity, dll.)
     * @return success Status keberhasilan reset
     */
    function resetState() external returns (bool success);

    /**
     * @dev Memvalidasi input untuk update parameter
     * @param parameters Array parameter
     * @param gradients Array gradien
     * @return isValid True jika input valid
     * @return errorMessage Pesan error jika input tidak valid
     */
    function validateInput(
        uint256[] calldata parameters,
        int256[] calldata gradients
    ) external pure returns (bool isValid, string memory errorMessage);

    /**
     * @dev Mengembalikan informasi tentang optimizer
     * @return name Nama optimizer
     * @return version Versi optimizer
     * @return optimizerType Tipe optimizer (SGD, Adam, RMSProp, dll.)
     * @return supportsMomentum True jika mendukung momentum
     * @return requiresGradientClipping True jika memerlukan gradient clipping
     */
    function getOptimizerInfo() external view returns (
        string memory name,
        string memory version,
        string memory optimizerType,
        bool supportsMomentum,
        bool requiresGradientClipping
    );

    /**
     * @dev Mengembalikan konfigurasi saat ini
     * @return learningRate Learning rate saat ini
     * @return momentum Momentum saat ini (jika tersedia)
     * @return beta1 Beta1 untuk Adam (jika tersedia)
     * @return beta2 Beta2 untuk Adam (jika tersedia)
     * @return epsilon Epsilon untuk stabilitas numerik
     */
    function getCurrentConfiguration() external view returns (
        uint256 learningRate,
        uint256 momentum,
        uint256 beta1,
        uint256 beta2,
        uint256 epsilon
    );

    /**
     * @dev Mengembalikan statistik penggunaan optimizer
     * @return totalUpdates Total jumlah update yang dilakukan
     * @return averageUpdateMagnitude Rata-rata besar update
     * @return lastUpdateTime Waktu update terakhir
     */
    function getOptimizerStats() external view returns (
        uint256 totalUpdates,
        uint256 averageUpdateMagnitude,
        uint256 lastUpdateTime
    );

    /**
     * @dev Clipping gradien untuk mencegah exploding gradients
     * @param gradients Array gradien
     * @param clipValue Nilai clipping maksimum
     * @return clippedGradients Array gradien yang sudah diclip
     */
    function clipGradients(
        int256[] calldata gradients,
        int256 clipValue
    ) external pure returns (int256[] memory clippedGradients);

    /**
     * @dev Normalisasi gradien untuk stabilitas training
     * @param gradients Array gradien
     * @return normalizedGradients Array gradien yang sudah dinormalisasi
     * @return normalizationFactor Faktor normalisasi yang digunakan
     */
    function normalizeGradients(
        int256[] calldata gradients
    ) external pure returns (int256[] memory normalizedGradients, uint256 normalizationFactor);
}