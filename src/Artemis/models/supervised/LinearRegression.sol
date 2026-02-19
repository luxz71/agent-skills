// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../../interfaces/IModel.sol";
import "../../loss/MSELoss.sol";
import "../../math/ArrayUtils.sol";
import "../../math/Statistics.sol";

/**
 * @title LinearRegression
 * @dev Implementasi Linear Regression model untuk supervised learning
 * @notice Model ini menggunakan gradient descent optimization dengan MSE loss function
 * @author Rizky Reza
 */
contract LinearRegression is IModel {
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    
    // Model parameters
    uint256[] private weights;
    uint256 private bias;
    
    // Training state
    bool private isTrained;
    uint256 private trainingEpochs;
    uint256 private currentLoss;
    
    // Hyperparameters
    uint256 private learningRate;
    uint256 private regularizationStrength;
    
    // Loss function instance
    MSELoss private mseLoss;
    
    /**
     * @dev Event khusus untuk Linear Regression
     * @param modelAddress Alamat contract model
     * @param weightsUpdated Array weights yang diupdate
     * @param biasUpdated Bias yang diupdate
     * @param epoch Epoch training saat ini
     */
    event ParametersUpdated(
        address indexed modelAddress,
        uint256[] weightsUpdated,
        uint256 biasUpdated,
        uint256 epoch
    );
    
    /**
     * @dev Constructor untuk Linear Regression
     * @param initialLearningRate Learning rate awal (dalam fixed-point)
     * @param initialRegularization Regularization strength (dalam fixed-point)
     */
    constructor(uint256 initialLearningRate, uint256 initialRegularization) {
        require(initialLearningRate > 0, "LinearRegression: learning rate must be positive");
        require(initialRegularization >= 0, "LinearRegression: regularization must be non-negative");
        
        learningRate = initialLearningRate;
        regularizationStrength = initialRegularization;
        isTrained = false;
        trainingEpochs = 0;
        currentLoss = 0;
        
        // Initialize MSELoss contract
        mseLoss = new MSELoss();
    }
    
    /**
     * @dev Melakukan training model dengan gradient descent
     * @param features Array 2D dari fitur training (baris x kolom)
     * @param labels Array 1D dari label training
     * @param epochs Jumlah epoch training
     * @param learningRateParam Learning rate untuk optimisasi
     * @return success Status keberhasilan training
     * @return finalLoss Nilai loss akhir setelah training
     */
    function train(
        uint256[][] calldata features,
        uint256[] calldata labels,
        uint256 epochs,
        uint256 learningRateParam
    ) external override returns (bool success, uint256 finalLoss) {
        require(features.length > 0, "LinearRegression: features cannot be empty");
        require(features.length == labels.length, "LinearRegression: features and labels must have same length");
        require(epochs > 0, "LinearRegression: epochs must be positive");
        require(learningRateParam > 0, "LinearRegression: learning rate must be positive");
        
        uint256 numFeatures = features[0].length;
        require(numFeatures > 0, "LinearRegression: features must have at least one column");
        
        // Initialize weights if not already initialized
        if (weights.length == 0) {
            _initializeWeights(numFeatures);
        } else {
            require(weights.length == numFeatures, "LinearRegression: feature dimension mismatch");
        }
        
        emit TrainingStarted(address(this), epochs, block.timestamp);
        
        learningRate = learningRateParam;
        uint256 bestLoss = type(uint256).max;
        uint256[] memory bestWeights = new uint256[](weights.length);
        uint256 bestBias = bias;
        
        for (uint256 epoch = 0; epoch < epochs; epoch++) {
            uint256 epochLoss = _trainEpoch(features, labels);
            currentLoss = epochLoss;
            trainingEpochs++;
            
            // Track best parameters
            if (epochLoss < bestLoss) {
                bestLoss = epochLoss;
                for (uint256 i = 0; i < weights.length; i++) {
                    bestWeights[i] = weights[i];
                }
                bestBias = bias;
            }
            
            emit EpochCompleted(address(this), epoch, epochLoss, 0);
            
            // Early stopping jika loss sudah konvergen
            if (epoch > 0 && _isConverged(epochLoss, currentLoss)) {
                break;
            }
        }
        
        // Restore best parameters
        for (uint256 i = 0; i < weights.length; i++) {
            weights[i] = bestWeights[i];
        }
        bias = bestBias;
        
        isTrained = true;
        finalLoss = currentLoss;
        
        emit TrainingCompleted(address(this), finalLoss, block.timestamp);
        
        return (true, finalLoss);
    }
    
    /**
     * @dev Melakukan prediksi menggunakan model linear
     * @param features Array 1D dari fitur untuk prediksi
     * @return prediction Hasil prediksi model (y = w0 + w1*x1 + ... + wn*xn)
     */
    function predict(uint256[] calldata features) external override view returns (uint256 prediction) {
        require(isTrained, "LinearRegression: model not trained");
        require(features.length == weights.length, "LinearRegression: feature dimension mismatch");
        
        prediction = bias;
        
        for (uint256 i = 0; i < features.length; i++) {
            prediction += (weights[i] * features[i]) / PRECISION;
        }
        
        return prediction;
    }
    
    /**
     * @dev Mengevaluasi performa model dengan data test
     * @param features Array 2D dari fitur test (baris x kolom)
     * @param labels Array 1D dari label test
     * @return accuracy R-squared score sebagai akurasi
     * @return loss Nilai MSE loss pada data test
     */
    function evaluate(
        uint256[][] calldata features,
        uint256[] calldata labels
    ) external override view returns (uint256 accuracy, uint256 loss) {
        require(isTrained, "LinearRegression: model not trained");
        require(features.length > 0, "LinearRegression: features cannot be empty");
        require(features.length == labels.length, "LinearRegression: features and labels must have same length");
        
        uint256[] memory predictions = new uint256[](features.length);
        uint256 totalSquaredError = 0;
        uint256 totalVariance = 0;
        
        // Hitung prediksi dan akumulasi error
        for (uint256 i = 0; i < features.length; i++) {
            predictions[i] = _predictSingle(features[i]);
            uint256 error;
            if (predictions[i] >= labels[i]) {
                error = predictions[i] - labels[i];
            } else {
                error = labels[i] - predictions[i];
            }
            totalSquaredError += (error * error) / PRECISION;
        }
        
        // Hitung R-squared
        uint256 meanLabel = ArrayUtils.mean(labels);
        for (uint256 i = 0; i < labels.length; i++) {
            uint256 diff;
            if (labels[i] >= meanLabel) {
                diff = labels[i] - meanLabel;
            } else {
                diff = meanLabel - labels[i];
            }
            totalVariance += (diff * diff) / PRECISION;
        }
        
        // MSE loss
        loss = totalSquaredError / features.length;
        
        // R-squared accuracy
        if (totalVariance == 0) {
            accuracy = PRECISION; // Perfect fit jika tidak ada variasi
        } else {
            uint256 rSquared = PRECISION - (totalSquaredError * PRECISION) / totalVariance;
            accuracy = rSquared > PRECISION ? PRECISION : (rSquared < 0 ? 0 : rSquared);
        }
        
        return (accuracy, loss);
    }
    
    /**
     * @dev Mengembalikan status training model
     * @return isTrained True jika model sudah ditraining
     * @return trainingEpochs Jumlah epoch yang sudah dijalankan
     * @return currentLoss Nilai loss terakhir
     */
    function getTrainingStatus() external override view returns (
        bool isTrained,
        uint256 trainingEpochs,
        uint256 currentLoss
    ) {
        return (isTrained, trainingEpochs, currentLoss);
    }
    
    /**
     * @dev Mengembalikan parameter model saat ini
     * @return parameters Array dari parameter model [bias, weights...]
     */
    function getParameters() external override view returns (uint256[] memory parameters) {
        parameters = new uint256[](weights.length + 1);
        parameters[0] = bias;
        for (uint256 i = 0; i < weights.length; i++) {
            parameters[i + 1] = weights[i];
        }
        return parameters;
    }
    
    /**
     * @dev Mengatur parameter model
     * @param parameters Array dari parameter baru [bias, weights...]
     * @return success Status keberhasilan pengaturan parameter
     */
    function setParameters(uint256[] calldata parameters) external override returns (bool success) {
        require(parameters.length >= 1, "LinearRegression: parameters must include at least bias");
        
        bias = parameters[0];
        weights = new uint256[](parameters.length - 1);
        
        for (uint256 i = 0; i < parameters.length - 1; i++) {
            weights[i] = parameters[i + 1];
        }
        
        isTrained = true;
        
        return true;
    }
    
    /**
     * @dev Mengembalikan metadata model
     * @return modelName "LinearRegression"
     * @return version "1.0.0"
     * @return inputSize Jumlah fitur input
     * @return outputSize 1 (regression output)
     */
    function getModelInfo() external override view returns (
        string memory modelName,
        string memory version,
        uint256 inputSize,
        uint256 outputSize
    ) {
        return (
            "LinearRegression",
            "1.0.0",
            weights.length,
            1
        );
    }
    
    /**
     * @dev Reset model ke kondisi awal (sebelum training)
     * @return success Status keberhasilan reset
     */
    function reset() external override returns (bool success) {
        if (weights.length > 0) {
            _initializeWeights(weights.length);
        }
        bias = 0;
        isTrained = false;
        trainingEpochs = 0;
        currentLoss = 0;
        
        return true;
    }
    
    /**
     * @dev Mengatur learning rate
     * @param newLearningRate Learning rate baru
     */
    function setLearningRate(uint256 newLearningRate) external {
        require(newLearningRate > 0, "LinearRegression: learning rate must be positive");
        learningRate = newLearningRate;
    }
    
    /**
     * @dev Mengatur regularization strength
     * @param newRegularization Regularization strength baru
     */
    function setRegularization(uint256 newRegularization) external {
        require(newRegularization >= 0, "LinearRegression: regularization must be non-negative");
        regularizationStrength = newRegularization;
    }
    
    /**
     * @dev Mengembalikan learning rate saat ini
     * @return currentLearningRate Learning rate saat ini
     */
    function getLearningRate() external view returns (uint256 currentLearningRate) {
        return learningRate;
    }
    
    /**
     * @dev Mengembalikan regularization strength saat ini
     * @return currentRegularization Regularization strength saat ini
     */
    function getRegularization() external view returns (uint256 currentRegularization) {
        return regularizationStrength;
    }
    
    /**
     * @dev Fungsi internal untuk training satu epoch
     * @param features Array fitur training
     * @param labels Array label training
     * @return epochLoss Nilai loss untuk epoch ini
     */
    function _trainEpoch(
        uint256[][] calldata features,
        uint256[] calldata labels
    ) internal returns (uint256 epochLoss) {
        uint256 numSamples = features.length;
        uint256[] memory weightGradients = new uint256[](weights.length);
        uint256 biasGradient = 0;
        uint256 totalLoss = 0;
        
        // Hitung gradients untuk setiap sample
        for (uint256 i = 0; i < numSamples; i++) {
            uint256 prediction = _predictSingle(features[i]);
            int256 errorGradient = mseLoss.calculateGradient(prediction, labels[i]);
            
            // Update bias gradient
            biasGradient += uint256(errorGradient);
            
            // Update weight gradients
            for (uint256 j = 0; j < weights.length; j++) {
                weightGradients[j] += (uint256(errorGradient) * features[i][j]) / PRECISION;
            }
            
            // Accumulate loss
            totalLoss += mseLoss.calculateLoss(prediction, labels[i]);
        }
        
        // Average gradients
        biasGradient = biasGradient / numSamples;
        for (uint256 j = 0; j < weights.length; j++) {
            weightGradients[j] = weightGradients[j] / numSamples;
        }
        
        // Apply regularization (L2 regularization)
        for (uint256 j = 0; j < weights.length; j++) {
            weightGradients[j] += (regularizationStrength * weights[j]) / PRECISION;
        }
        
        // Update parameters dengan gradient descent
        bias = _updateParameter(bias, biasGradient);
        for (uint256 j = 0; j < weights.length; j++) {
            weights[j] = _updateParameter(weights[j], weightGradients[j]);
        }
        
        emit ParametersUpdated(address(this), weights, bias, trainingEpochs);
        
        return totalLoss / numSamples;
    }
    
    /**
     * @dev Fungsi internal untuk prediksi single sample
     * @param sample Array fitur untuk satu sample
     * @return prediction Hasil prediksi
     */
    function _predictSingle(uint256[] memory sample) internal view returns (uint256 prediction) {
        prediction = bias;
        for (uint256 i = 0; i < weights.length; i++) {
            prediction += (weights[i] * sample[i]) / PRECISION;
        }
        return prediction;
    }
    
    /**
     * @dev Fungsi internal untuk update parameter dengan gradient descent
     * @param parameter Parameter saat ini
     * @param gradient Gradien parameter
     * @return updatedParameter Parameter yang sudah diupdate
     */
    function _updateParameter(uint256 parameter, uint256 gradient) internal view returns (uint256 updatedParameter) {
        int256 signedParameter = int256(parameter);
        int256 signedGradient = int256(gradient);
        int256 update = (signedGradient * int256(learningRate)) / int256(PRECISION);
        
        if (signedParameter - update < 0) {
            updatedParameter = 0;
        } else {
            updatedParameter = uint256(signedParameter - update);
        }
        
        return updatedParameter;
    }
    
    /**
     * @dev Fungsi internal untuk inisialisasi weights
     * @param numFeatures Jumlah fitur
     */
    function _initializeWeights(uint256 numFeatures) internal {
        weights = new uint256[](numFeatures);
        bias = 0;
        
        // Initialize weights dengan nilai kecil random (dalam konteks blockchain, gunakan nilai deterministik)
        for (uint256 i = 0; i < numFeatures; i++) {
            weights[i] = PRECISION / 100; // 0.01
        }
    }
    
    /**
     * @dev Fungsi internal untuk mengecek konvergensi
     * @param previousLoss Loss sebelumnya
     * @param currentLoss Loss saat ini
     * @return converged True jika sudah konvergen
     */
    function _isConverged(uint256 previousLoss, uint256 currentLoss) internal pure returns (bool converged) {
        if (previousLoss == 0) return false;
        
        uint256 lossChange;
        if (currentLoss >= previousLoss) {
            lossChange = currentLoss - previousLoss;
        } else {
            lossChange = previousLoss - currentLoss;
        }
        
        uint256 relativeChange = (lossChange * PRECISION) / previousLoss;
        
        // Konvergen jika perubahan relatif kurang dari 0.1%
        return relativeChange < (PRECISION / 1000);
    }
}