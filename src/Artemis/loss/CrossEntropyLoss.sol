// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../interfaces/ILossFunction.sol";

/**
 * @title CrossEntropyLoss
 * @dev Implementasi Cross Entropy Loss function untuk classification
 * @notice Fungsi loss yang menghitung cross entropy antara predicted probabilities dan true labels
 * @author Rizky Reza
 */
contract CrossEntropyLoss is ILossFunction {
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    uint256 private constant EPSILON = 1e12; // Small value untuk mencegah log(0)
    
    /**
     * @dev Event khusus untuk Cross Entropy Loss
     * @param prediction Prediksi model (probability)
     * @param target Target yang sebenarnya (0 atau 1 untuk binary, one-hot untuk multi-class)
     * @param loss Nilai loss yang dihitung
     */
    event CrossEntropyLossCalculated(uint256 prediction, uint256 target, uint256 loss);
    
    /**
     * @dev Menghitung Cross Entropy loss untuk single sample
     * @param prediction Prediksi model (probability)
     * @param target Target yang sebenarnya (0 atau 1)
     * @return loss Nilai loss untuk sample ini
     */
    function calculateLoss(
        uint256 prediction,
        uint256 target
    ) external override pure returns (uint256 loss) {
        return _calculateLoss(prediction, target);
    }
    
    /**
     * @dev Menghitung Cross Entropy loss untuk batch data
     * @param predictions Array prediksi model (probabilities)
     * @param targets Array target yang sebenarnya (0 atau 1)
     * @return totalLoss Total loss untuk batch
     * @return averageLoss Rata-rata loss untuk batch
     */
    function calculateBatchLoss(
        uint256[] calldata predictions,
        uint256[] calldata targets
    ) external override pure returns (uint256 totalLoss, uint256 averageLoss) {
        require(predictions.length == targets.length, "CrossEntropyLoss: Predictions and targets must have same length");
        require(predictions.length > 0, "CrossEntropyLoss: Input arrays cannot be empty");
        
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
     * @dev Menghitung gradien Cross Entropy loss terhadap prediksi
     * @param prediction Prediksi model (probability)
     * @param target Target yang sebenarnya (0 atau 1)
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
     * @param predictions Array prediksi model (probabilities)
     * @param targets Array target yang sebenarnya (0 atau 1)
     * @return gradients Array gradien untuk setiap sample
     * @return averageGradient Rata-rata gradien
     */
    function calculateBatchGradient(
        uint256[] calldata predictions,
        uint256[] calldata targets
    ) external override pure returns (int256[] memory gradients, int256 averageGradient) {
        require(predictions.length == targets.length, "CrossEntropyLoss: Predictions and targets must have same length");
        require(predictions.length > 0, "CrossEntropyLoss: Input arrays cannot be empty");
        
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
            return (false, "CrossEntropyLoss: Predictions and targets must have same length");
        }
        
        if (predictions.length == 0) {
            return (false, "CrossEntropyLoss: Input arrays cannot be empty");
        }
        
        // Validasi bahwa predictions adalah probabilities (antara 0 dan 1)
        for (uint256 i = 0; i < predictions.length; i++) {
            if (predictions[i] > ONE) {
                return (false, "CrossEntropyLoss: Predictions must be probabilities between 0 and 1");
            }
        }
        
        // Validasi bahwa targets adalah 0 atau 1 (untuk binary classification)
        for (uint256 i = 0; i < targets.length; i++) {
            if (targets[i] != 0 && targets[i] != ONE) {
                return (false, "CrossEntropyLoss: Targets must be 0 or 1 for binary classification");
            }
        }
        
        return (true, "");
    }
    
    /**
     * @dev Mengembalikan informasi tentang loss function
     * @return name "CrossEntropy"
     * @return version "1.0.0"
     * @return isDifferentiable true
     * @return rangeMin 0
     * @return rangeMax type(uint256).max / 10 (untuk mencegah overflow)
     */
    function getLossFunctionInfo() external override pure returns (
        string memory name,
        string memory version,
        bool isDifferentiable,
        uint256 rangeMin,
        uint256 rangeMax
    ) {
        return (
            "CrossEntropy",
            "1.0.0",
            true,
            0,
            type(uint256).max / 10
        );
    }
    
    /**
     * @dev Mengembalikan konfigurasi loss function
     * @return supportsBatch true
     * @return requiresNormalization true (predictions harus probabilities)
     * @return precision PRECISION
     */
    function getConfiguration() external override pure returns (
        bool supportsBatch,
        bool requiresNormalization,
        uint256 precision
    ) {
        return (true, true, PRECISION);
    }
    
    /**
     * @dev Fungsi internal untuk menghitung Cross Entropy loss
     * @param prediction Prediksi model (probability)
     * @param target Target yang sebenarnya (0 atau 1)
     * @return loss Nilai loss
     */
    function _calculateLoss(uint256 prediction, uint256 target) internal pure returns (uint256 loss) {
        // Cross Entropy: -[target * log(prediction) + (1 - target) * log(1 - prediction)]
        
        // Clipping prediction untuk mencegah log(0)
        uint256 clippedPrediction = _clipProbability(prediction);
        uint256 clippedOneMinusPrediction = _clipProbability(ONE - prediction);
        
        int256 term1;
        int256 term2;
        
        if (target == ONE) {
            // Jika target = 1: -log(prediction)
            term1 = int256(_log(clippedPrediction));
            loss = uint256(-term1); // Negative log likelihood
        } else if (target == 0) {
            // Jika target = 0: -log(1 - prediction)
            term2 = int256(_log(clippedOneMinusPrediction));
            loss = uint256(-term2); // Negative log likelihood
        } else {
            // Untuk multi-class, target harus one-hot encoded
            revert("CrossEntropyLoss: Target must be 0 or 1 for binary classification");
        }
        
        return loss;
    }
    
    /**
     * @dev Fungsi internal untuk menghitung gradien Cross Entropy
     * @param prediction Prediksi model (probability)
     * @param target Target yang sebenarnya (0 atau 1)
     * @return gradient Gradien loss
     */
    function _calculateGradient(uint256 prediction, uint256 target) internal pure returns (int256 gradient) {
        // Gradient Cross Entropy: (prediction - target) / [prediction * (1 - prediction)]
        // Untuk numerical stability, gunakan simplified version: (prediction - target)
        
        if (target == ONE) {
            // Jika target = 1: gradient = (prediction - 1) / prediction
            // Simplified: gradient = prediction - target
            gradient = int256(prediction) - int256(target);
        } else if (target == 0) {
            // Jika target = 0: gradient = prediction / (1 - prediction)
            // Simplified: gradient = prediction - target
            gradient = int256(prediction) - int256(target);
        } else {
            revert("CrossEntropyLoss: Target must be 0 or 1 for binary classification");
        }
        
        // Scale dengan precision
        gradient = gradient / int256(PRECISION);
        
        return gradient;
    }
    
    /**
     * @dev Fungsi internal untuk clipping probability
     * @param probability Nilai probability
     * @return clippedProbability Nilai probability yang sudah di-clip
     */
    function _clipProbability(uint256 probability) internal pure returns (uint256 clippedProbability) {
        if (probability < EPSILON) {
            return EPSILON;
        } else if (probability > (ONE - EPSILON)) {
            return ONE - EPSILON;
        } else {
            return probability;
        }
    }
    
    /**
     * @dev Fungsi internal untuk menghitung natural logarithm dengan fixed-point
     * @param x Nilai input dalam fixed-point
     * @return result ln(x) dalam fixed-point
     */
    function _log(uint256 x) internal pure returns (uint256 result) {
        require(x > 0, "CrossEntropyLoss: Cannot take log of zero");
        
        // Approximation sederhana untuk ln(x) menggunakan Taylor series di sekitar 1
        // ln(x) ≈ (x - 1) - (x - 1)^2/2 + (x - 1)^3/3 - ...
        
        if (x == ONE) {
            return 0;
        }
        
        // Untuk nilai mendekati 0, return nilai positif besar (akan dihandle di calculateLoss)
        if (x < EPSILON) {
            return 40 * PRECISION; // |ln(epsilon)| ≈ 40
        }
        
        // Untuk nilai besar, gunakan scaling
        if (x > 100 * PRECISION) {
            return 5 * PRECISION; // Approximation untuk nilai besar
        }
        
        // Simple approximation: ln(x) ≈ 2 * (x - 1) / (x + 1) untuk x mendekati 1
        int256 xMinusOne = int256(x) - int256(ONE);
        int256 xPlusOne = int256(x) + int256(ONE);
        
        if (xPlusOne == 0) {
            return 40 * PRECISION; // |ln(epsilon)| ≈ 40
        }
        
        int256 logApprox = (2 * xMinusOne * int256(PRECISION)) / xPlusOne;
        
        if (logApprox < 0) {
            return uint256(-logApprox);
        } else {
            return uint256(logApprox);
        }
    }
}