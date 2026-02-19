
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../../interfaces/IModel.sol";
import "../../math/ArrayUtils.sol";
import "../../math/Statistics.sol";

/**
 * @title KNearestNeighbors
 * @dev Implementasi K-Nearest Neighbors model untuk classification dan regression
 * @notice Model non-parametric instance-based learning dengan multiple distance metrics
 * @author Rizky Reza
 */
contract KNearestNeighbors is IModel {
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    
    // Training data storage
    uint256[][] private trainingFeatures;
    uint256[] private trainingLabels;
    
    // Model configuration
    uint256 private kValue;
    DistanceMetric private distanceMetric;
    WeightingStrategy private weightingStrategy;
    ModelType private modelType;
    
    // Training state
    bool private isTrained;
    uint256 private trainingEpochs;
    uint256 private currentLoss;
    
    // Enums for configuration
    enum DistanceMetric {
        Euclidean,
        Manhattan,
        Minkowski
    }
    
    enum WeightingStrategy {
        Uniform,
        Distance
    }
    
    enum ModelType {
        Classification,
        Regression
    }
    
    /**
     * @dev Event khusus untuk KNN
     * @param modelAddress Alamat contract model
     * @param kValue K value yang digunakan
     * @param distanceMetric Metric jarak yang digunakan
     * @param numSamples Jumlah training samples
     */
    event KNNConfigured(
        address indexed modelAddress,
        uint256 kValue,
        DistanceMetric distanceMetric,
        uint256 numSamples
    );
    
    /**
     * @dev Event untuk neighbor selection
     * @param modelAddress Alamat contract model
     * @param queryPoint Titik query
     * @param neighbors Indeks neighbors terpilih
     * @param distances Jarak ke neighbors
     */
    event NeighborsSelected(
        address indexed modelAddress,
        uint256[] queryPoint,
        uint256[] neighbors,
        uint256[] distances
    );
    
    /**
     * @dev Constructor untuk K-Nearest Neighbors
     * @param initialK K value awal
     * @param initialMetric Metric jarak awal
     * @param initialWeighting Strategy weighting awal
     * @param initialType Tipe model (Classification/Regression)
     */
    constructor(
        uint256 initialK,
        DistanceMetric initialMetric,
        WeightingStrategy initialWeighting,
        ModelType initialType
    ) {
        require(initialK > 0, "KNearestNeighbors: k must be positive");
        
        kValue = initialK;
        distanceMetric = initialMetric;
        weightingStrategy = initialWeighting;
        modelType = initialType;
        isTrained = false;
        trainingEpochs = 0;
        currentLoss = 0;
    }
    
    /**
     * @dev Melakukan training model (menyimpan data training)
     * @param features Array 2D dari fitur training (baris x kolom)
     * @param labels Array 1D dari label training
     * @param epochs Tidak digunakan untuk KNN (selalu 1)
     * @param learningRate Tidak digunakan untuk KNN (selalu 0)
     * @return success Status keberhasilan training
     * @return finalLoss Selalu 0 untuk KNN
     */
    function train(
        uint256[][] calldata features,
        uint256[] calldata labels,
        uint256 epochs,
        uint256 learningRate
    ) external override returns (bool success, uint256 finalLoss) {
        require(features.length > 0, "KNearestNeighbors: features cannot be empty");
        require(features.length == labels.length, "KNearestNeighbors: features and labels must have same length");
        require(features.length >= kValue, "KNearestNeighbors: k must be <= number of samples");
        
        uint256 numFeatures = features[0].length;
        require(numFeatures > 0, "KNearestNeighbors: features must have at least one column");
        
        emit TrainingStarted(address(this), 1, block.timestamp);
        
        // Store training data
        trainingFeatures = new uint256[][](features.length);
        trainingLabels = new uint256[](labels.length);
        
        for (uint256 i = 0; i < features.length; i++) {
            trainingFeatures[i] = features[i];
            trainingLabels[i] = labels[i];
        }
        
        isTrained = true;
        trainingEpochs = 1;
        currentLoss = 0;
        
        emit KNNConfigured(address(this), kValue, distanceMetric, features.length);
        emit TrainingCompleted(address(this), 0, block.timestamp);
        
        return (true, 0);
    }
    
    /**
     * @dev Melakukan prediksi menggunakan KNN
     * @param features Array 1D dari fitur untuk prediksi
     * @return prediction Hasil prediksi (kelas untuk classification, nilai untuk regression)
     */
    function predict(uint256[] calldata features) external override view returns (uint256 prediction) {
        require(isTrained, "KNearestNeighbors: model not trained");
        require(features.length == trainingFeatures[0].length, "KNearestNeighbors: feature dimension mismatch");
        
        // Find k nearest neighbors
        (uint256[] memory neighborIndices, uint256[] memory distances) = _findNearestNeighbors(features);
        
        if (modelType == ModelType.Classification) {
            prediction = _classify(neighborIndices, distances);
        } else {
            prediction = _regress(neighborIndices, distances);
        }
        
        return prediction;
    }
    
    /**
     * @dev Mengevaluasi performa model dengan data test
     * @param features Array 2D dari fitur test (baris x kolom)
     * @param labels Array 1D dari label test
     * @return accuracy Akurasi untuk classification, R-squared untuk regression
     * @return loss Nilai loss (0-1 untuk classification, MSE untuk regression)
     */
    function evaluate(
        uint256[][] calldata features,
        uint256[] calldata labels
    ) external override view returns (uint256 accuracy, uint256 loss) {
        require(isTrained, "KNearestNeighbors: model not trained");
        require(features.length > 0, "KNearestNeighbors: features cannot be empty");
        require(features.length == labels.length, "KNearestNeighbors: features and labels must have same length");
        
        if (modelType == ModelType.Classification) {
            return _evaluateClassification(features, labels);
        } else {
            return _evaluateRegression(features, labels);
        }
    }
    
    /**
     * @dev Mengembalikan status training model
     * @return isTrained True jika model sudah ditraining
     * @return trainingEpochs Selalu 1 untuk KNN
     * @return currentLoss Selalu 0 untuk KNN
     */
    function getTrainingStatus() external override view returns (
        bool isTrained,
        uint256 trainingEpochs,
        uint256 currentLoss
    ) {
        return (isTrained, trainingEpochs, currentLoss);
    }
    
    /**
     * @dev Mengembalikan parameter model (tidak berlaku untuk KNN)
     * @return parameters Array kosong (KNN tidak memiliki parameters)
     */
    function getParameters() external override view returns (uint256[] memory parameters) {
        return new uint256[](0);
    }
    
    /**
     * @dev Mengatur parameter model (tidak berlaku untuk KNN)
     * @param parameters Tidak digunakan
     * @return success Selalu true
     */
    function setParameters(uint256[] calldata parameters) external override returns (bool success) {
        // KNN doesn't have trainable parameters
        return true;
    }
    
    /**
     * @dev Mengembalikan metadata model
     * @return modelName "KNearestNeighbors"
     * @return version "1.0.0"
     * @return inputSize Jumlah fitur input
     * @return outputSize 1 (single output)
     */
    function getModelInfo() external override view returns (
        string memory modelName,
        string memory version,
        uint256 inputSize,
        uint256 outputSize
    ) {
        uint256 featureSize = isTrained && trainingFeatures.length > 0 ? trainingFeatures[0].length : 0;
        return (
            "KNearestNeighbors",
            "1.0.0",
            featureSize,
            1
        );
    }
    
    /**
     * @dev Reset model ke kondisi awal (sebelum training)
     * @return success Status keberhasilan reset
     */
    function reset() external override returns (bool success) {
        delete trainingFeatures;
        delete trainingLabels;
        isTrained = false;
        trainingEpochs = 0;
        currentLoss = 0;
        
        return true;
    }
    
    /**
     * @dev Mengatur K value
     * @param newK K value baru
     */
    function setKValue(uint256 newK) external {
        require(newK > 0, "KNearestNeighbors: k must be positive");
        require(!isTrained || newK <= trainingFeatures.length, "KNearestNeighbors: k must be <= number of samples");
        kValue = newK;
    }
    
    /**
     * @dev Mengatur distance metric
     * @param newMetric Metric jarak baru
     */
    function setDistanceMetric(DistanceMetric newMetric) external {
        distanceMetric = newMetric;
    }
    
    /**
     * @dev Mengatur weighting strategy
     * @param newWeighting Strategy weighting baru
     */
    function setWeightingStrategy(WeightingStrategy newWeighting) external {
        weightingStrategy = newWeighting;
    }
    
    /**
     * @dev Mengatur model type
     * @param newType Tipe model baru
     */
    function setModelType(ModelType newType) external {
        modelType = newType;
    }
    
    /**
     * @dev Mengembalikan K value saat ini
     * @return currentK K value saat ini
     */
    function getKValue() external view returns (uint256 currentK) {
        return kValue;
    }
    
    /**
     * @dev Mengembalikan distance metric saat ini
     * @return currentMetric Distance metric saat ini
     */
    function getDistanceMetric() external view returns (DistanceMetric currentMetric) {
        return distanceMetric;
    }
    
    /**
     * @dev Mengembalikan weighting strategy saat ini
     * @return currentWeighting Weighting strategy saat ini
     */
    function getWeightingStrategy() external view returns (WeightingStrategy currentWeighting) {
        return weightingStrategy;
    }
    
    /**
     * @dev Mengembalikan model type saat ini
     * @return currentType Model type saat ini
     */
    function getModelType() external view returns (ModelType currentType) {
        return modelType;
    }
    
    /**
     * @dev Mengembalikan jumlah training samples
     * @return numSamples Jumlah training samples
     */
    function getNumTrainingSamples() external view returns (uint256 numSamples) {
        return trainingFeatures.length;
    }
    
    /**
     * @dev Fungsi internal untuk mencari k nearest neighbors
     * @param queryPoint Titik query
     * @return neighborIndices Indeks neighbors terpilih
     * @return distances Jarak ke neighbors
     */
    function _findNearestNeighbors(
        uint256[] memory queryPoint
    ) internal view returns (uint256[] memory neighborIndices, uint256[] memory distances) {
        uint256 numSamples = trainingFeatures.length;
        uint256[] memory allDistances = new uint256[](numSamples);
        uint256[] memory allIndices = new uint256[](numSamples);
        
        // Hitung jarak ke semua training samples
        for (uint256 i = 0; i < numSamples; i++) {
            allDistances[i] = _calculateDistance(queryPoint, trainingFeatures[i]);
            allIndices[i] = i;
        }
        
        // Sort berdasarkan jarak (selection sort sederhana)
        for (uint256 i = 0; i < kValue; i++) {
            uint256 minIndex = i;
            for (uint256 j = i + 1; j < numSamples; j++) {
                if (allDistances[j] < allDistances[minIndex]) {
                    minIndex = j;
                }
            }
            
            // Swap
            if (minIndex != i) {
                (allDistances[i], allDistances[minIndex]) = (allDistances[minIndex], allDistances[i]);
                (allIndices[i], allIndices[minIndex]) = (allIndices[minIndex], allIndices[i]);
            }
        }
        
        // Ambil k terdekat
        neighborIndices = new uint256[](kValue);
        distances = new uint256[](kValue);
        
        for (uint256 i = 0; i < kValue; i++) {
            neighborIndices[i] = allIndices[i];
            distances[i] = allDistances[i];
        }
        
        return (neighborIndices, distances);
    }
    
    /**
     * @dev Fungsi internal untuk classification
     * @param neighborIndices Indeks neighbors
     * @param distances Jarak ke neighbors
     * @return prediction Kelas prediksi
     */
    function _classify(
        uint256[] memory neighborIndices,
        uint256[] memory distances
    ) internal view returns (uint256 prediction) {
        if (weightingStrategy == WeightingStrategy.Uniform) {
            // Majority voting
            return _majorityVote(neighborIndices);
        } else {
            // Distance-weighted voting
            return _weightedVote(neighborIndices, distances);
        }
    }
    
    /**
     * @dev Fungsi internal untuk regression
     * @param neighborIndices Indeks neighbors
     * @param distances Jarak ke neighbors
     * @return prediction Nilai prediksi
     */
    function _regress(
        uint256[] memory neighborIndices,
        uint256[] memory distances
    ) internal view returns (uint256 prediction) {
        if (weightingStrategy == WeightingStrategy.Uniform) {
            // Simple average
            uint256 sum = 0;
            for (uint256 i = 0; i < neighborIndices.length; i++) {
                sum += trainingLabels[neighborIndices[i]];
            }
            return sum / neighborIndices.length;
        } else {
            // Distance-weighted average
            uint256 weightedSum = 0;
            uint256 weightSum = 0;
            
            for (uint256 i = 0; i < neighborIndices.length; i++) {
                uint256 weight = distances[i] > 0 ? (PRECISION * PRECISION) / distances[i] : PRECISION * 10;
                weightedSum += (trainingLabels[neighborIndices[i]] * weight) / PRECISION;
                weightSum += weight;
            }
            
            return weightSum > 0 ? (weightedSum * PRECISION) / weightSum : 0;
        }
    }
    
    /**
     * @dev Fungsi internal untuk majority voting
     * @param neighborIndices Indeks neighbors
     * @return majorityClass Kelas mayoritas
     */
    function _majorityVote(uint256[] memory neighborIndices) internal view returns (uint256 majorityClass) {
        // Untuk implementasi sederhana, gunakan array untuk menyimpan class yang unik
        uint256[] memory uniqueClasses = new uint256[](neighborIndices.length);
        uint256[] memory voteCounts = new uint256[](neighborIndices.length);
        uint256 uniqueCount = 0;
        
        // Count votes for each class
        for (uint256 i = 0; i < neighborIndices.length; i++) {
            uint256 label = trainingLabels[neighborIndices[i]];
            bool found = false;
            
            // Cari apakah class sudah ada di uniqueClasses
            for (uint256 j = 0; j < uniqueCount; j++) {
                if (uniqueClasses[j] == label) {
                    voteCounts[j]++;
                    found = true;
                    break;
                }
            }
            
            // Jika belum ada, tambahkan ke uniqueClasses
            if (!found) {
                uniqueClasses[uniqueCount] = label;
                voteCounts[uniqueCount] = 1;
                uniqueCount++;
            }
        }
        
        // Find class with maximum votes
        uint256 maxVotes = 0;
        for (uint256 i = 0; i < uniqueCount; i++) {
            if (voteCounts[i] > maxVotes) {
                maxVotes = voteCounts[i];
                majorityClass = uniqueClasses[i];
            }
        }
        
        return majorityClass;
    }
    
    /**
     * @dev Fungsi internal untuk weighted voting
     * @param neighborIndices Indeks neighbors
     * @param distances Jarak ke neighbors
     * @return weightedClass Kelas dengan voting tertimbang tertinggi
     */
    function _weightedVote(
        uint256[] memory neighborIndices,
        uint256[] memory distances
    ) internal view returns (uint256 weightedClass) {
        // Untuk implementasi sederhana, gunakan array untuk menyimpan class yang unik
        uint256[] memory uniqueClasses = new uint256[](neighborIndices.length);
        uint256[] memory weightedVotes = new uint256[](neighborIndices.length);
        uint256 uniqueCount = 0;
        
        // Calculate weighted votes for each class
        for (uint256 i = 0; i < neighborIndices.length; i++) {
            uint256 label = trainingLabels[neighborIndices[i]];
            uint256 weight = distances[i] > 0 ? (PRECISION * PRECISION) / distances[i] : PRECISION * 10;
            bool found = false;
            
            // Cari apakah class sudah ada di uniqueClasses
            for (uint256 j = 0; j < uniqueCount; j++) {
                if (uniqueClasses[j] == label) {
                    weightedVotes[j] += weight;
                    found = true;
                    break;
                }
            }
            
            // Jika belum ada, tambahkan ke uniqueClasses
            if (!found) {
                uniqueClasses[uniqueCount] = label;
                weightedVotes[uniqueCount] = weight;
                uniqueCount++;
            }
        }
        
        // Find class with maximum weighted votes
        uint256 maxWeightedVotes = 0;
        for (uint256 i = 0; i < uniqueCount; i++) {
            if (weightedVotes[i] > maxWeightedVotes) {
                maxWeightedVotes = weightedVotes[i];
                weightedClass = uniqueClasses[i];
            }
        }
        
        return weightedClass;
    }
    
    /**
     * @dev Fungsi internal untuk menghitung jarak
     * @param point1 Titik pertama
     * @param point2 Titik kedua
     * @return distance Jarak antara dua titik
     */
    function _calculateDistance(
        uint256[] memory point1,
        uint256[] memory point2
    ) internal view returns (uint256 distance) {
        require(point1.length == point2.length, "KNearestNeighbors: points must have same dimension");
        
        if (distanceMetric == DistanceMetric.Euclidean) {
            return _euclideanDistance(point1, point2);
        } else if (distanceMetric == DistanceMetric.Manhattan) {
            return _manhattanDistance(point1, point2);
        } else {
            return _minkowskiDistance(point1, point2, 3); // p=3 untuk Minkowski
        }
    }
    
    /**
     * @dev Fungsi internal untuk Euclidean distance
     * @param point1 Titik pertama
     * @param point2 Titik kedua
     * @return distance Euclidean distance
     */
    function _euclideanDistance(
        uint256[] memory point1,
        uint256[] memory point2
    ) internal pure returns (uint256 distance) {
        uint256 sumSquared = 0;
        for (uint256 i = 0; i < point1.length; i++) {
            uint256 diff;
            if (point1[i] >= point2[i]) {
                diff = point1[i] - point2[i];
            } else {
                diff = point2[i] - point1[i];
            }
            sumSquared += (diff * diff) / PRECISION;
        }
        return ArrayUtils.sqrt(sumSquared);
    }
    
    /**
     * @dev Fungsi internal untuk Manhattan distance
     * @param point1 Titik pertama
     * @param point2 Titik kedua
     * @return distance Manhattan distance
     */
    function _manhattanDistance(
        uint256[] memory point1,
        uint256[] memory point2
    ) internal pure returns (uint256 distance) {
        uint256 sum = 0;
        for (uint256 i = 0; i < point1.length; i++) {
            if (point1[i] >= point2[i]) {
                sum += point1[i] - point2[i];
            } else {
                sum += point2[i] - point1[i];
            }
        }
        return sum;
    }
    
    /**
     * @dev Fungsi internal untuk Minkowski distance
     * @param point1 Titik pertama
     * @param point2 Titik kedua
     * @param p Parameter p untuk Minkowski
     * @return distance Minkowski distance
     */
    function _minkowskiDistance(
        uint256[] memory point1,
        uint256[] memory point2,
        uint256 p
    ) internal pure returns (uint256 distance) {
        uint256 sum = 0;
        for (uint256 i = 0; i < point1.length; i++) {
            uint256 diff;
            if (point1[i] >= point2[i]) {
                diff = point1[i] - point2[i];
            } else {
                diff = point2[i] - point1[i];
            }
            uint256 poweredDiff = _power(diff, p);
            sum += poweredDiff;
        }
        return _power(sum, PRECISION / p); // p-th root
    }
    
    /**
     * @dev Fungsi internal untuk evaluasi classification
     * @param features Array fitur test
     * @param labels Array label test
     * @return accuracy Akurasi
     * @return loss Loss (1 - accuracy)
     */
    function _evaluateClassification(
        uint256[][] calldata features,
        uint256[] calldata labels
    ) internal view returns (uint256 accuracy, uint256 loss) {
        uint256 correctPredictions = 0;
        
        for (uint256 i = 0; i < features.length; i++) {
            uint256 prediction = _predictSingle(features[i]);
            if (prediction == labels[i]) {
                correctPredictions++;
            }
        }
        
        accuracy = (correctPredictions * PRECISION) / features.length;
        loss = PRECISION - accuracy;
        
        return (accuracy, loss);
    }
    
    /**
     * @dev Fungsi internal untuk evaluasi regression
     * @param features Array fitur test
     * @param labels Array label test
     * @return accuracy R-squared
     * @return loss MSE loss
     */
    function _evaluateRegression(
        uint256[][] calldata features,
        uint256[] calldata labels
    ) internal view returns (uint256 accuracy, uint256 loss) {
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
     * @dev Fungsi internal untuk prediksi single sample
     * @param features Array fitur untuk prediksi
     * @return prediction Hasil prediksi
     */
    function _predictSingle(uint256[] memory features) internal view returns (uint256 prediction) {
        // Find k nearest neighbors
        (uint256[] memory neighborIndices, uint256[] memory distances) = _findNearestNeighbors(features);
        
        if (modelType == ModelType.Classification) {
            prediction = _classify(neighborIndices, distances);
        } else {
            prediction = _regress(neighborIndices, distances);
        }
        
        return prediction;
    }
    
    /**
     * @dev Fungsi internal untuk menghitung power
     * @param base Basis
     * @param exponent Eksponen (dalam fixed-point)
     * @return result Hasil perpangkatan
     */
    function _power(uint256 base, uint256 exponent) internal pure returns (uint256 result) {
        // Simple approximation untuk fixed-point power
        if (exponent == PRECISION) {
            return base;
        } else if (exponent == 0) {
            return PRECISION;
        } else if (exponent == PRECISION / 2) {
            return ArrayUtils.sqrt(base);
        }
        
        // Untuk implementasi sederhana, gunakan approximation
        return base; // Untuk sekarang return base sebagai placeholder
    }
}