// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title IModel
 * @dev Interface dasar untuk semua model machine learning dalam Artemis
 * @notice Interface ini mendefinisikan kontrak yang harus diimplementasikan oleh semua model ML
 * @author Rizky Reza
 */
interface IModel {
    /**
     * @dev Event yang dipancarkan ketika training model dimulai
     * @param modelAddress Alamat contract model
     * @param epochCount Jumlah epoch yang akan dijalankan
     * @param timestamp Waktu training dimulai
     */
    event TrainingStarted(
        address indexed modelAddress,
        uint256 epochCount,
        uint256 timestamp
    );

    /**
     * @dev Event yang dipancarkan ketika training model selesai
     * @param modelAddress Alamat contract model
     * @param finalLoss Nilai loss akhir setelah training
     * @param timestamp Waktu training selesai
     */
    event TrainingCompleted(
        address indexed modelAddress,
        uint256 finalLoss,
        uint256 timestamp
    );

    /**
     * @dev Event yang dipancarkan pada setiap epoch training
     * @param modelAddress Alamat contract model
     * @param epoch Epoch saat ini
     * @param loss Nilai loss pada epoch ini
     * @param accuracy Nilai akurasi pada epoch ini (jika tersedia)
     */
    event EpochCompleted(
        address indexed modelAddress,
        uint256 epoch,
        uint256 loss,
        uint256 accuracy
    );

    /**
     * @dev Melakukan training model dengan data yang diberikan
     * @param features Array 2D dari fitur training (baris x kolom)
     * @param labels Array 1D dari label training
     * @param epochs Jumlah epoch training
     * @param learningRate Learning rate untuk optimisasi
     * @return success Status keberhasilan training
     * @return finalLoss Nilai loss akhir setelah training
     */
    function train(
        uint256[][] calldata features,
        uint256[] calldata labels,
        uint256 epochs,
        uint256 learningRate
    ) external returns (bool success, uint256 finalLoss);

    /**
     * @dev Melakukan prediksi menggunakan model yang sudah ditraining
     * @param features Array 1D dari fitur untuk prediksi
     * @return prediction Hasil prediksi model
     */
    function predict(uint256[] calldata features) external view returns (uint256 prediction);

    /**
     * @dev Mengevaluasi performa model dengan data test
     * @param features Array 2D dari fitur test (baris x kolom)
     * @param labels Array 1D dari label test
     * @return accuracy Akurasi model pada data test
     * @return loss Nilai loss pada data test
     */
    function evaluate(
        uint256[][] calldata features,
        uint256[] calldata labels
    ) external view returns (uint256 accuracy, uint256 loss);

    /**
     * @dev Mengembalikan status training model
     * @return isTrained True jika model sudah ditraining
     * @return trainingEpochs Jumlah epoch yang sudah dijalankan
     * @return currentLoss Nilai loss terakhir
     */
    function getTrainingStatus() external view returns (
        bool isTrained,
        uint256 trainingEpochs,
        uint256 currentLoss
    );

    /**
     * @dev Mengembalikan parameter model saat ini
     * @return parameters Array dari parameter model
     */
    function getParameters() external view returns (uint256[] memory parameters);

    /**
     * @dev Mengatur parameter model
     * @param parameters Array dari parameter baru
     * @return success Status keberhasilan pengaturan parameter
     */
    function setParameters(uint256[] calldata parameters) external returns (bool success);

    /**
     * @dev Mengembalikan metadata model
     * @return modelName Nama model
     * @return version Versi model
     * @return inputSize Ukuran input model
     * @return outputSize Ukuran output model
     */
    function getModelInfo() external view returns (
        string memory modelName,
        string memory version,
        uint256 inputSize,
        uint256 outputSize
    );

    /**
     * @dev Reset model ke kondisi awal (sebelum training)
     * @return success Status keberhasilan reset
     */
    function reset() external returns (bool success);
}