
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../../interfaces/IModel.sol";
import "../../loss/CrossEntropyLoss.sol";
import "../../activation/Sigmoid.sol";
import "../../math/ArrayUtils.sol";
import "../../math/Statistics.sol";

/**
 * @title LogisticRegression
 * @dev Implementasi Logistic Regression model untuk binary classification
 * @notice Model ini menggunakan gradient descent optimization dengan Cross Entropy Loss dan Sigmoid activation
 * @author Rizky Reza
 */
contract LogisticRegression is IModel {
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    uint256 private constant THRESHOLD = 5e17; // 0.5 untuk binary classification
    
    // Model parameters
    uint256[] private weights;
    uint256 private bias;
    
    // Training state
    bool private isTrained;
    uint256 private trainingEpochs;
    uint256 private currentLoss;
    uint256 private currentAccuracy;
    
    // Hyperparameters
    uint256 private learningRate;
    uint256 private regularizationStrength;
    
    // Component instances
    CrossEntropyLoss private crossEntropyLoss;
    Sigmoid private sigmoid;
    
    /**
     * @dev Event khusus untuk Logistic Regression
     * @param modelAddress Alamat contract model
     * @param weightsUpdated Array weights yang diupdate
     * @param biasUpdated Bias yang diupdate
     * @param epoch Epoch training saat ini
     * @param accuracy Akurasi pada epoch ini
     */
    event ParametersUpdated(
        address indexed modelAddress,
        uint256[] weightsUpdated,
        uint256 biasUpdated,
        uint256 epoch,
        uint256 accuracy
    );
    
    /**
     * @dev Event untuk classification metrics
     * @param modelAddress Alamat contract model
     * @param truePositives True Positives
     * @param falsePositives False Positives
     * @param trueNegatives True Negatives
     * @param falseNegatives False Negatives
     */
    event ClassificationMetrics(
        address indexed modelAddress,
        uint256 truePositives,
        uint256 falsePositives,
        uint256 trueNegatives,
        uint256 falseNegatives
    );
    
    /**
     * @dev Constructor untuk Logistic Regression
     * @param initialLearningRate Learning rate awal (dalam fixed-point)
     * @param initialRegularization Regularization strength (dalam fixed-point)
     */
    constructor(uint256 initialLearningRate, uint256 initialRegularization) {
        require(initialLearningRate > 0, "LogisticRegression: learning rate must be positive");
        require(initialRegularization >= 0, "LogisticRegression: regularization must be non-negative");
        
        learningRate = initialLearningRate;
        regularizationStrength = initialRegularization;
        isTrained = false;
        trainingEpochs = 0;
        currentLoss = 0;
        currentAccuracy = 0;
        
        // Initialize component contracts
        crossEntropyLoss = new CrossEntropyLoss();
        sigmoid = new Sigmoid();
    }
    
    /**
     * @dev Melakukan training model dengan gradient descent
     * @param features Array 2D dari fitur training (baris x kolom)
     * @param labels Array 1D dari label training (0 atau 1)
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
        require(features.length > 0, "LogisticRegression: features cannot be empty");
        require(features.length == labels.length, "LogisticRegression: features and labels must have same length");
        require(epochs > 0, "LogisticRegression: epochs must be positive");
        require(learningRateParam > 0, "LogisticRegression: learning rate must be positive");
        
        // Validate labels (must be 0 or 1)
        for (uint256 i = 0; i < labels.length; i++) {
            require(labels[i] == 0 || labels[i] == ONE, "LogisticRegression: labels must be 0 or 1");
        }
        
        uint256 numFeatures = features[0].length;
        require(numFeatures > 0, "LogisticRegression: features must have at least one column");
        
        // Initialize weights if not already initialized
        if (weights.length == 0) {
            _initializeWeights(numFeatures);
        } else {
            require(weights.length == numFeatures, "LogisticRegression: feature dimension mismatch");
        }
        
        emit TrainingStarted(address(this), epochs, block.timestamp);
        
        learningRate = learningRateParam;
        uint256 bestLoss = type(uint256).max;
        uint256 bestAccuracy = 0;
        uint256[] memory bestWeights = new uint256[](weights.length);
        uint256 bestBias = bias;
        
        for (uint256 epoch = 0; epoch < epochs; epoch++) {
            (uint256 epochLoss, uint256 epochAccuracy) = _trainEpoch(features, labels);
            currentLoss = epochLoss;
            currentAccuracy = epochAccuracy;
            trainingEpochs++;
            
            // Track best parameters
            if (epochLoss < bestLoss || (epochLoss == bestLoss && epochAccuracy > bestAccuracy)) {
                bestLoss = epochLoss;
                bestAccuracy = epochAccuracy;
                for (uint256 i = 0; i < weights.length; i++) {
                    bestWeights[i] = weights[i];
                }
                bestBias = bias;
            }
            
            emit EpochCompleted(address(this), epoch, epochLoss, epochAccuracy);
            emit ParametersUpdated(address(this), weights, bias, epoch, epochAccuracy);
            
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
     * @dev Melakukan prediksi menggunakan model logistic regression
     * @param features Array 1D dari fitur untuk prediksi
     * @return prediction Probability kelas positif (dalam fixed-point, 0-1)
     */
    function predict(uint256[] calldata features) external override view returns (uint256 prediction) {
        require(isTrained, "LogisticRegression: model not trained");
        require(features.length == weights.length, "LogisticRegression: feature dimension mismatch");
        
        // Linear combination: z = bias + w1*x1 + ... + wn*xn
        uint256 z = bias;
        for (uint256 i = 0; i < features.length; i++) {
            z += (weights[i] * features[i]) / PRECISION;
        }
        
        // Apply sigmoid activation: P(y=1) = 1 / (1 + e^(-z))
        prediction = sigmoid.activate(z);
        
        return prediction;
    }
    
    /**
     * @dev Melakukan binary classification dengan threshold
     * @param features Array 1D dari fitur untuk prediksi
     * @return class Kelas prediksi (0 atau 1)
     */
    function classify(uint256[] calldata features) external view returns (uint256 class) {
        uint256 probability = _predictSingle(features);
        return probability >= THRESHOLD ? ONE : 0;
    }
    
    /**
     * @dev Mengevaluasi performa model dengan data test
     * @param features Array 2D dari fitur test (baris x kolom)
     * @param labels Array 1D dari label test (0 atau 1)
     * @return accuracy Akurasi model pada data test
     * @return loss Nilai Cross Entropy loss pada data test
     */
    function evaluate(
        uint256[][] calldata features,
        uint256[] calldata labels
    ) external override view returns (uint256 accuracy, uint256 loss) {
        require(isTrained, "LogisticRegression: model not trained");
        require(features.length > 0, "LogisticRegression: features cannot be empty");
        require(features.length == labels.length, "LogisticRegression: features and labels must have same length");
        
        uint256 correctPredictions = 0;
        uint256 totalLoss = 0;
        uint256 truePositives = 0;
        uint256 falsePositives = 0;
        uint256 trueNegatives = 0;
        uint256 falseNegatives = 0;
        
        for (uint256 i = 0; i < features.length; i++) {
            uint256 prediction = _predictSingle(features[i]);
            uint256 predictedClass = prediction >= THRESHOLD ? ONE : 0;
            
            // Calculate loss
            totalLoss += crossEntropyLoss.calculateLoss(prediction, labels[i]);
            
            // Calculate accuracy and confusion matrix
            if (predictedClass == labels[i]) {
                correctPredictions++;
                if (predictedClass == ONE) {
                    truePositives++;
                } else {
                    trueNegatives++;
                }
            } else {
                if (predictedClass == ONE) {
                    falsePositives++;
                } else {
                    falseNegatives++;
                }
            }
        }
        
        accuracy = (correctPredictions * PRECISION) / features.length;
        loss = totalLoss / features.length;
        
        return (accuracy, loss);
    }
    
    /**
     * @dev Mengembalikan classification metrics detail
     * @param features Array 2D dari fitur test
     * @param labels Array 1D dari label test
     * @return precision Precision score
     * @return recall Recall score
     * @return f1Score F1-score
     * @return accuracy Akurasi
     */
    function getClassificationMetrics(
        uint256[][] calldata features,
        uint256[] calldata labels
    ) external view returns (
        uint256 precision,
        uint256 recall,
        uint256 f1Score,
        uint256 accuracy
    ) {
        require(isTrained, "LogisticRegression: model not trained");
        require(features.length > 0, "LogisticRegression: features cannot be empty");
        require(features.length == labels.length, "LogisticRegression: features and labels must have same length");
        
        uint256 truePositives = 0;
        uint256 falsePositives = 0;
        uint256 falseNegatives = 0;
        uint256 correctPredictions = 0;
        
        for (uint256 i = 0; i < features.length; i++) {
            uint256 prediction = _predictSingle(features[i]);
            uint256 predictedClass = prediction >= THRESHOLD ? ONE : 0;
            
            if (predictedClass == labels[i]) {
                correctPredictions++;
                if (predictedClass == ONE) {
                    truePositives++;
                }
            } else {
                if (predictedClass == ONE) {
                    falsePositives++;
                } else {
                    falseNegatives++;
                }
            }
        }
        
        accuracy = (correctPredictions * PRECISION) / features.length;
        
        // Calculate precision
        uint256 predictedPositives = truePositives + falsePositives;
        if (predictedPositives > 0) {
            precision = (truePositives * PRECISION) / predictedPositives;
        } else {
            precision = 0;
        }
        
        // Calculate recall
        uint256 actualPositives = truePositives + falseNegatives;
        if (actualPositives > 0) {
            recall = (truePositives * PRECISION) / actualPositives;
        } else {
            recall = 0;
        }
        
        // Calculate F1-score
        if (precision + recall > 0) {
            f1Score = (2 * precision * recall * PRECISION) / ((precision + recall) * PRECISION);
        } else {
            f1Score = 0;
        }
        
        return (precision, recall, f1Score, accuracy);
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
        require(parameters.length >= 1, "LogisticRegression: parameters must include at least bias");
        
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
     * @return modelName "LogisticRegression"
     * @return version "1.0.0"
     * @return inputSize Jumlah fitur input
     * @return outputSize 1 (probability output)
     */
    function getModelInfo() external override view returns (
        string memory modelName,
        string memory version,
        uint256 inputSize,
        uint256 outputSize
    ) {
        return (
            "LogisticRegression",
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
        currentAccuracy = 0;
        
        return true;
    }
    
    /**
     * @dev Mengatur learning rate
     * @param newLearningRate Learning rate baru
     */
    function setLearningRate(uint256 newLearningRate) external {
        require(newLearningRate > 0, "LogisticRegression: learning rate must be positive");
        learningRate = newLearningRate;
    }
    
    /**
     * @dev Mengatur regularization strength
     * @param newRegularization Regularization strength baru
     */
    function setRegularization(uint256 newRegularization) external {
        require(newRegularization >= 0, "LogisticRegression: regularization must be non-negative");
        regularizationStrength = newRegularization;
    }
    
    /**
     * @dev Mengatur classification threshold
     * @param newThreshold Threshold baru (0-1 dalam fixed-point)
     */
    function setThreshold(uint256 newThreshold) external {
        require(newThreshold >= 0 && newThreshold <= PRECISION, "LogisticRegression: threshold must be between 0 and 1");
        // Note: THRESHOLD is constant, this would require contract modification
        // For now, we'll use the constant threshold
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
     * @dev Mengembalikan akurasi terakhir
     * @return lastAccuracy Akurasi terakhir dari training
     */
    function getLastAccuracy() external view returns (uint256 lastAccuracy) {
        return currentAccuracy;
    }
    
    /**
     * @dev Fungsi internal untuk training satu epoch
     * @param features Array fitur training
     * @param labels Array label training
     * @return epochLoss Nilai loss untuk epoch ini
     * @return epochAccuracy Akurasi untuk epoch ini
     */
    function _trainEpoch(
        uint256[][] calldata features,
        uint256[] calldata labels
    ) internal returns (uint256 epochLoss, uint256 epochAccuracy) {
        uint256 numSamples = features.length;
        uint256[] memory weightGradients = new uint256[](weights.length);
        uint256 biasGradient = 0;
        uint256 totalLoss = 0;
        uint256 correctPredictions = 0;
        
        // Hitung gradients untuk setiap sample
        for (uint256 i = 0; i < numSamples; i++) {
            uint256 prediction = _predictSingle(features[i]);
            int256 errorGradient = crossEntropyLoss.calculateGradient(prediction, labels[i]);
            
            // Update bias gradient
            biasGradient += uint256(errorGradient);
            
            // Update weight gradients
            for (uint256 j = 0; j < weights.length; j++) {
                weightGradients[j] += (uint256(errorGradient) * features[i][j]) / PRECISION;
            }
            
            // Accumulate loss
            totalLoss += crossEntropyLoss.calculateLoss(prediction, labels[i]);
            
            // Track accuracy
            uint256 predictedClass = prediction >= THRESHOLD ? ONE : 0;
            if (predictedClass == labels[i]) {
                correctPredictions++;
            }
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
        
        epochLoss = totalLoss / numSamples;
        epochAccuracy = (correctPredictions * PRECISION) / numSamples;
        
        return (epochLoss, epochAccuracy);
    }
    
    /**
     * @dev Fungsi internal untuk prediksi single sample
     * @param sample Array fitur untuk satu sample
     * @return prediction Hasil prediksi (probability)
     */
    function _predictSingle(uint256[] memory sample) internal view returns (uint256 prediction) {
        // Linear combination: z = bias + w1*x1 + ... + wn*xn
        uint256 z = bias;
        for (uint256 i = 0; i < weights.length; i++) {
            z += (weights[i] * sample[i]) / PRECISION;
        }
        
        // Apply sigmoid activation
        prediction = sigmoid.activate(z);
        
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
        
        // Initialize weights dengan nilai kecil random
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