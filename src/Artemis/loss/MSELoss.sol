// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../interfaces/ILossFunction.sol";

/**
 * @title MSELoss
 * @dev Implementasi Mean Squared Error loss function untuk regression
 * @notice Fungsi loss yang menghitung rata-rata kuadrat error antara prediksi dan target
 * @author Rizky Reza
 */
contract MSELoss is ILossFunction {
    uint256 private constant PRECISION = 1e18;
    uint256 private constant TWO = 2 * 1e18;
    
    /**
     * @dev Event khusus untuk MSE Loss
     * @param prediction Prediksi model
     * @param target Target yang sebenarnya
     * @param loss Nilai loss yang dihitung
     */
    event MSELossCalculated(uint256 prediction, uint256 target, uint256 loss);
    
    /**
     * @dev Menghitung MSE loss untuk single sample
     * @param prediction Prediksi model
     * @param target Target yang sebenarnya
     * @return loss Nilai loss untuk sample ini
     */
    function calculateLoss(
        uint256 prediction,
        uint256 target
    ) external override pure returns (uint256 loss) {
        return _calculateLoss(prediction, target);
    }
    
    /**
     * @dev Menghitung MSE loss untuk batch data
     * @param predictions Array prediksi model
     * @param targets Array target yang sebenarnya
     * @return totalLoss Total loss untuk batch
     * @return averageLoss Rata-rata loss untuk batch
     */
    function calculateBatchLoss(
        uint256[] calldata predictions,
        uint256[] calldata targets
    ) external override pure returns (uint256 totalLoss, uint256 averageLoss) {
        require(predictions.length == targets.length, "MSELoss: Predictions and targets must have same length");
        require(predictions.length > 0, "MSELoss: Input arrays cannot be empty");
        
        uint256 batchSize = predictions.length;
        totalLoss = 0;
        
        for (uint256 i = 0; i < batchSize; i++) {
            uint256 loss = _calculateLoss(predictions[i], targets[i]);
            totalLoss += loss;
        }
        
        averageLoss = totalLoss / batchSize;
        
        return (totalLoss, averageLoss);
    }
    
    /**
     * @dev Menghitung gradien MSE loss terhadap prediksi
     * @param prediction Prediksi model
     * @param target Target yang sebenarnya
     * @return gradient Gradien loss terhadap prediksi
     */
    function calculateGradient(
        uint256 prediction,
        uint256 target
    ) external override pure returns (int256 gradient) {
        return _calculateGradient(prediction, target);
    }
    
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
    ) external override pure returns (int256[] memory gradients, int256 averageGradient) {
        require(predictions.length == targets.length, "MSELoss: Predictions and targets must have same length");
        require(predictions.length > 0, "MSELoss: Input arrays cannot be empty");
        
        uint256 batchSize = predictions.length;
        gradients = new int256[](batchSize);
        int256 sumGradient = 0;
        
        for (uint256 i = 0; i < batchSize; i++) {
            gradients[i] = _calculateGradient(predictions[i], targets[i]);
            sumGradient += gradients[i];
        }
        
        averageGradient = sumGradient / int256(batchSize);
        
        return (gradients, averageGradient);
    }
    
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
    ) external override pure returns (bool isValid, string memory errorMessage) {
        if (predictions.length != targets.length) {
            return (false, "MSELoss: Predictions and targets must have same length");
        }
        
        if (predictions.length == 0) {
            return (false, "MSELoss: Input arrays cannot be empty");
        }
        
        return (true, "");
    }
    
    /**
     * @dev Mengembalikan informasi tentang loss function
     * @return name "MSE"
     * @return version "1.0.0"
     * @return isDifferentiable true
     * @return rangeMin 0
     * @return rangeMax type(uint256).max
     */
    function getLossFunctionInfo() external override pure returns (
        string memory name,
        string memory version,
        bool isDifferentiable,
        uint256 rangeMin,
        uint256 rangeMax
    ) {
        return (
            "MSE",
            "1.0.0",
            true,
            0,
            type(uint256).max
        );
    }
    
    /**
     * @dev Mengembalikan konfigurasi loss function
     * @return supportsBatch true
     * @return requiresNormalization false
     * @return precision PRECISION
     */
    function getConfiguration() external override pure returns (
        bool supportsBatch,
        bool requiresNormalization,
        uint256 precision
    ) {
        return (true, false, PRECISION);
    }
    
    /**
     * @dev Fungsi internal untuk menghitung MSE loss
     * @param prediction Prediksi model
     * @param target Target yang sebenarnya
     * @return loss Nilai loss
     */
    function _calculateLoss(uint256 prediction, uint256 target) internal pure returns (uint256 loss) {
        // MSE: (prediction - target)^2
        uint256 error;
        if (prediction >= target) {
            error = prediction - target;
        } else {
            error = target - prediction;
        }
        
        loss = (error * error) / PRECISION;
        
        return loss;
    }
    
    /**
     * @dev Fungsi internal untuk menghitung gradien MSE
     * @param prediction Prediksi model
     * @param target Target yang sebenarnya
     * @return gradient Gradien loss
     */
    function _calculateGradient(uint256 prediction, uint256 target) internal pure returns (int256 gradient) {
        // Gradient MSE: 2 * (prediction - target)
        if (prediction >= target) {
            gradient = int256(2 * (prediction - target));
        } else {
            gradient = -int256(2 * (target - prediction));
        }
        
        // Scale dengan precision
        gradient = gradient / int256(PRECISION);
        
        return gradient;
    }
}