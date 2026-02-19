
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../../interfaces/IModel.sol";
import "../../math/ArrayUtils.sol";
import "../../math/Matrix.sol";
import "../../math/Statistics.sol";

/**
 * @title KMeans
 * @dev Implementasi K-Means clustering model untuk unsupervised learning
 * @notice Model ini menggunakan algoritma expectation-maximization dengan multiple centroid initialization methods
 * @author Rizky Reza
 */
contract KMeans is IModel {
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    
    // Enum untuk centroid initialization methods
    enum InitializationMethod {
        RANDOM,
        KMEANS_PLUS_PLUS,
        MANUAL
    }
    
    // Model parameters
    uint256[][] private centroids; // [k][features]
    uint256[] private clusterAssignments; // [samples]
    uint256[] private clusterSizes; // [k]
    
    // Training state
    bool private isTrained;
    uint256 private trainingIterations;
    uint256 private currentWCSS;
    
    // Hyperparameters
    uint256 private k; // Number of clusters
    uint256 private maxIterations;
    uint256 private convergenceThreshold;
    InitializationMethod private initMethod;
    
    // Evaluation metrics
    uint256 private silhouetteScore;
    uint256 private daviesBouldinIndex;
    
    /**
     * @dev Event khusus untuk K-Means clustering
     * @param modelAddress Alamat contract model
     * @param iteration Iterasi training saat ini
     * @param wcss Within-cluster sum of squares
     * @param centroidMovement Total movement centroids
     */
    event ClusteringProgress(
        address indexed modelAddress,
        uint256 iteration,
        uint256 wcss,
        uint256 centroidMovement
    );
    
    /**
     * @dev Event untuk centroid initialization
     * @param modelAddress Alamat contract model
     * @param method Metode inisialisasi yang digunakan
     * @param centroidsInitialized Centroids yang diinisialisasi
     */
    event CentroidsInitialized(
        address indexed modelAddress,
        InitializationMethod method,
        uint256[][] centroidsInitialized
    );
    
    /**
     * @dev Event untuk cluster assignment
     * @param modelAddress Alamat contract model
     * @param assignments Cluster assignments untuk semua samples
     * @param clusterSizes Ukuran setiap cluster
     */
    event ClustersAssigned(
        address indexed modelAddress,
        uint256[] assignments,
        uint256[] clusterSizes
    );
    
    /**
     * @dev Constructor untuk K-Means clustering
     * @param numClusters Jumlah cluster (K)
     * @param maxIters Maksimum iterasi training
     * @param threshold Threshold konvergensi
     * @param initializationMethod Metode inisialisasi centroid
     */
    constructor(
        uint256 numClusters,
        uint256 maxIters,
        uint256 threshold,
        InitializationMethod initializationMethod
    ) {
        require(numClusters > 0, "KMeans: number of clusters must be positive");
        require(maxIters > 0, "KMeans: max iterations must be positive");
        require(threshold > 0, "KMeans: convergence threshold must be positive");
        
        k = numClusters;
        maxIterations = maxIters;
        convergenceThreshold = threshold;
        initMethod = initializationMethod;
        
        isTrained = false;
        trainingIterations = 0;
        currentWCSS = 0;
        silhouetteScore = 0;
        daviesBouldinIndex = 0;
    }
    
    /**
     * @dev Melakukan training model K-Means clustering
     * @param features Array 2D dari fitur training (baris x kolom)
     * @param labels Tidak digunakan untuk unsupervised learning (harus empty)
     * @param epochs Jumlah epoch training (tidak digunakan, gunakan maxIterations)
     * @param learningRate Tidak digunakan untuk K-Means (harus 0)
     * @return success Status keberhasilan training
     * @return finalLoss Nilai final WCSS setelah training
     */
    function train(
        uint256[][] calldata features,
        uint256[] calldata labels,
        uint256 epochs,
        uint256 learningRate
    ) external override returns (bool success, uint256 finalLoss) {
        require(features.length > 0, "KMeans: features cannot be empty");
        require(features[0].length > 0, "KMeans: features must have at least one dimension");
        require(labels.length == 0, "KMeans: labels should be empty for unsupervised learning");
        require(learningRate == 0, "KMeans: learning rate not used for K-Means");
        
        uint256 numSamples = features.length;
        uint256 numFeatures = features[0].length;
        
        // Initialize centroids berdasarkan metode yang dipilih
        _initializeCentroids(features, numFeatures);
        
        emit TrainingStarted(address(this), maxIterations, block.timestamp);
        
        bool converged = false;
        uint256 iteration = 0;
        uint256 bestWCSS = type(uint256).max;
        uint256[][] memory bestCentroids = new uint256[][](k);
        uint256[] memory bestAssignments = new uint256[](numSamples);
        
        for (iteration = 0; iteration < maxIterations && !converged; iteration++) {
            // Expectation step: assign samples to nearest centroids
            uint256[] memory newAssignments = _assignClusters(features);
            
            // Maximization step: update centroids
            uint256 totalMovement = _updateCentroids(features, newAssignments);
            
            // Calculate WCSS (Within-Cluster Sum of Squares)
            uint256 wcss = _calculateWCSS(features, newAssignments);
            currentWCSS = wcss;
            
            // Track best parameters
            if (wcss < bestWCSS) {
                bestWCSS = wcss;
                for (uint256 i = 0; i < k; i++) {
                    bestCentroids[i] = new uint256[](numFeatures);
                    for (uint256 j = 0; j < numFeatures; j++) {
                        bestCentroids[i][j] = centroids[i][j];
                    }
                }
                for (uint256 i = 0; i < numSamples; i++) {
                    bestAssignments[i] = newAssignments[i];
                }
            }
            
            emit ClusteringProgress(address(this), iteration, wcss, totalMovement);
            emit EpochCompleted(address(this), iteration, wcss, 0);
            
            // Check convergence
            converged = totalMovement < convergenceThreshold || iteration == maxIterations - 1;
        }
        
        // Restore best parameters
        for (uint256 i = 0; i < k; i++) {
            for (uint256 j = 0; j < numFeatures; j++) {
                centroids[i][j] = bestCentroids[i][j];
            }
        }
        clusterAssignments = bestAssignments;
        
        // Calculate evaluation metrics
        silhouetteScore = _calculateSilhouetteScore(features, clusterAssignments);
        daviesBouldinIndex = _calculateDaviesBouldinIndex(features, clusterAssignments);
        
        isTrained = true;
        trainingIterations = iteration;
        finalLoss = bestWCSS;
        
        emit TrainingCompleted(address(this), finalLoss, block.timestamp);
        
        return (true, finalLoss);
    }
    
    /**
     * @dev Melakukan prediksi cluster untuk data baru
     * @param features Array 1D dari fitur untuk prediksi
     * @return prediction Cluster assignment (0 hingga k-1)
     */
    function predict(uint256[] calldata features) external override view returns (uint256 prediction) {
        require(isTrained, "KMeans: model not trained");
        require(features.length == centroids[0].length, "KMeans: feature dimension mismatch");
        
        uint256 minDistance = type(uint256).max;
        prediction = 0;
        
        for (uint256 i = 0; i < k; i++) {
            uint256 distance = _euclideanDistance(features, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                prediction = i;
            }
        }
        
        return prediction;
    }
    
    /**
     * @dev Mengevaluasi performa clustering (tidak menggunakan labels)
     * @param features Array 2D dari fitur test (baris x kolom)
     * @param labels Tidak digunakan untuk unsupervised learning (harus empty)
     * @return accuracy Silhouette score sebagai akurasi
     * @return loss WCSS sebagai loss
     */
    function evaluate(
        uint256[][] calldata features,
        uint256[] calldata labels
    ) external override view returns (uint256 accuracy, uint256 loss) {
        require(isTrained, "KMeans: model not trained");
        require(features.length > 0, "KMeans: features cannot be empty");
        require(labels.length == 0, "KMeans: labels should be empty for unsupervised evaluation");
        
        // Predict clusters untuk data test menggunakan fungsi internal
        uint256[] memory testAssignments = new uint256[](features.length);
        for (uint256 i = 0; i < features.length; i++) {
            testAssignments[i] = _predictSingle(features[i]);
        }
        
        // Calculate metrics
        uint256 testWCSS = _calculateWCSS(features, testAssignments);
        uint256 testSilhouette = _calculateSilhouetteScore(features, testAssignments);
        
        return (testSilhouette, testWCSS);
    }
    
    /**
     * @dev Mengembalikan status training model
     * @return isTrained True jika model sudah ditraining
     * @return trainingIterations Jumlah iterasi yang sudah dijalankan
     * @return currentWCSS Nilai WCSS terakhir
     */
    function getTrainingStatus() external override view returns (
        bool isTrained,
        uint256 trainingIterations,
        uint256 currentWCSS
    ) {
        return (isTrained, trainingIterations, currentWCSS);
    }
    
    /**
     * @dev Mengembalikan parameter model (centroids flattened)
     * @return parameters Array dari parameter model (centroids dalam row-major order)
     */
    function getParameters() external override view returns (uint256[] memory parameters) {
        require(isTrained, "KMeans: model not trained");
        
        uint256 numFeatures = centroids[0].length;
        parameters = new uint256[](k * numFeatures);
        
        uint256 index = 0;
        for (uint256 i = 0; i < k; i++) {
            for (uint256 j = 0; j < numFeatures; j++) {
                parameters[index++] = centroids[i][j];
            }
        }
        
        return parameters;
    }
    
    /**
     * @dev Mengatur parameter model (centroids)
     * @param parameters Array dari parameter baru (centroids dalam row-major order)
     * @return success Status keberhasilan pengaturan parameter
     */
    function setParameters(uint256[] calldata parameters) external override returns (bool success) {
        require(parameters.length > 0, "KMeans: parameters cannot be empty");
        require(parameters.length % k == 0, "KMeans: parameters length must be divisible by k");
        
        uint256 numFeatures = parameters.length / k;
        centroids = new uint256[][](k);
        
        uint256 index = 0;
        for (uint256 i = 0; i < k; i++) {
            centroids[i] = new uint256[](numFeatures);
            for (uint256 j = 0; j < numFeatures; j++) {
                centroids[i][j] = parameters[index++];
            }
        }
        
        isTrained = true;
        return true;
    }
    
    /**
     * @dev Mengembalikan metadata model
     * @return modelName "KMeans"
     * @return version "1.0.0"
     * @return inputSize Jumlah fitur input
     * @return outputSize Jumlah cluster (k)
     */
    function getModelInfo() external override view returns (
        string memory modelName,
        string memory version,
        uint256 inputSize,
        uint256 outputSize
    ) {
        uint256 inputDim = isTrained ? centroids[0].length : 0;
        return (
            "KMeans",
            "1.0.0",
            inputDim,
            k
        );
    }
    
    /**
     * @dev Reset model ke kondisi awal (sebelum training)
     * @return success Status keberhasilan reset
     */
    function reset() external override returns (bool success) {
        delete centroids;
        delete clusterAssignments;
        delete clusterSizes;
        
        isTrained = false;
        trainingIterations = 0;
        currentWCSS = 0;
        silhouetteScore = 0;
        daviesBouldinIndex = 0;
        
        return true;
    }
    
    /**
     * @dev Mengembalikan centroids saat ini
     * @return currentCentroids Array 2D dari centroids
     */
    function getCentroids() external view returns (uint256[][] memory currentCentroids) {
        require(isTrained, "KMeans: model not trained");
        return centroids;
    }
    
    /**
     * @dev Mengembalikan cluster assignments saat ini
     * @return assignments Array cluster assignments
     */
    function getClusterAssignments() external view returns (uint256[] memory assignments) {
        require(isTrained, "KMeans: model not trained");
        return clusterAssignments;
    }
    
    /**
     * @dev Mengembalikan evaluation metrics
     * @return wcss Within-cluster sum of squares
     * @return silhouette Silhouette score
     * @return dbIndex Davies-Bouldin index
     */
    function getEvaluationMetrics() external view returns (
        uint256 wcss,
        uint256 silhouette,
        uint256 dbIndex
    ) {
        require(isTrained, "KMeans: model not trained");
        return (currentWCSS, silhouetteScore, daviesBouldinIndex);
    }
    
    // ========== INTERNAL FUNCTIONS ==========
    
    /**
     * @dev Fungsi internal untuk prediksi single sample
     * @param sample Array fitur untuk satu sample
     * @return prediction Hasil prediksi cluster
     */
    function _predictSingle(uint256[] memory sample) internal view returns (uint256 prediction) {
        uint256 minDistance = type(uint256).max;
        prediction = 0;
        
        for (uint256 i = 0; i < k; i++) {
            uint256 distance = _euclideanDistance(sample, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                prediction = i;
            }
        }
        
        return prediction;
    }
    
    /**
     * @dev Inisialisasi centroids berdasarkan metode yang dipilih
     * @param features Data training
     * @param numFeatures Jumlah fitur
     */
    function _initializeCentroids(
        uint256[][] calldata features,
        uint256 numFeatures
    ) internal {
        centroids = new uint256[][](k);
        
        if (initMethod == InitializationMethod.RANDOM) {
            _initializeRandom(features, numFeatures);
        } else if (initMethod == InitializationMethod.KMEANS_PLUS_PLUS) {
            _initializeKMeansPlusPlus(features, numFeatures);
        } else {
            // MANUAL - akan di-set melalui setParameters
            for (uint256 i = 0; i < k; i++) {
                centroids[i] = new uint256[](numFeatures);
                for (uint256 j = 0; j < numFeatures; j++) {
                    centroids[i][j] = 0;
                }
            }
        }
        
        emit CentroidsInitialized(address(this), initMethod, centroids);
    }
    
    /**
     * @dev Random centroid initialization
     */
    function _initializeRandom(
        uint256[][] calldata features,
        uint256 numFeatures
    ) internal {
        uint256 numSamples = features.length;
        
        for (uint256 i = 0; i < k; i++) {
            uint256 randomIndex = uint256(keccak256(abi.encodePacked(block.timestamp, i))) % numSamples;
            centroids[i] = new uint256[](numFeatures);
            for (uint256 j = 0; j < numFeatures; j++) {
                centroids[i][j] = features[randomIndex][j];
            }
        }
    }
    
    /**
     * @dev K-means++ centroid initialization
     */
    function _initializeKMeansPlusPlus(
        uint256[][] calldata features,
        uint256 numFeatures
    ) internal {
        uint256 numSamples = features.length;
        uint256[] memory distances = new uint256[](numSamples);
        uint256 totalDistance = 0;
        
        // Pilih centroid pertama secara random
        uint256 firstIndex = uint256(keccak256(abi.encodePacked(block.timestamp))) % numSamples;
        centroids[0] = new uint256[](numFeatures);
        for (uint256 j = 0; j < numFeatures; j++) {
            centroids[0][j] = features[firstIndex][j];
        }
        
        // Pilih centroid selanjutnya dengan probability proportional to squared distance
        for (uint256 i = 1; i < k; i++) {
            totalDistance = 0;
            
            // Hitung jarak setiap titik ke centroid terdekat
            for (uint256 s = 0; s < numSamples; s++) {
                uint256 minDist = type(uint256).max;
                for (uint256 c = 0; c < i; c++) {
                    uint256 dist = _euclideanDistance(features[s], centroids[c]);
                    if (dist < minDist) {
                        minDist = dist;
                    }
                }
                distances[s] = minDist * minDist / PRECISION; // squared distance
                totalDistance += distances[s];
            }
            
            // Pilih centroid berikutnya dengan probability proportional to squared distance
            uint256 randomValue = uint256(keccak256(abi.encodePacked(block.timestamp, i))) % totalDistance;
            uint256 cumulative = 0;
            
            for (uint256 s = 0; s < numSamples; s++) {
                cumulative += distances[s];
                if (cumulative >= randomValue) {
                    centroids[i] = new uint256[](numFeatures);
                    for (uint256 j = 0; j < numFeatures; j++) {
                        centroids[i][j] = features[s][j];
                    }
                    break;
                }
            }
        }
    }
    
    /**
     * @dev Assign clusters untuk semua samples
     * @param features Data training
     * @return assignments Array cluster assignments
     */
    function _assignClusters(
        uint256[][] calldata features
    ) internal returns (uint256[] memory assignments) {
        uint256 numSamples = features.length;
        assignments = new uint256[](numSamples);
        clusterSizes = new uint256[](k);
        
        for (uint256 i = 0; i < numSamples; i++) {
            uint256 minDistance = type(uint256).max;
            uint256 bestCluster = 0;
            
            for (uint256 j = 0; j < k; j++) {
                uint256 distance = _euclideanDistance(features[i], centroids[j]);
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = j;
                }
            }
            
            assignments[i] = bestCluster;
            clusterSizes[bestCluster]++;
        }
        
        emit ClustersAssigned(address(this), assignments, clusterSizes);
        return assignments;
    }
    
    /**
     * @dev Update centroids berdasarkan cluster assignments
     * @param features Data training
     * @param assignments Cluster assignments
     * @return totalMovement Total pergerakan centroids
     */
    function _updateCentroids(
        uint256[][] calldata features,
        uint256[] memory assignments
    ) internal returns (uint256 totalMovement) {
        uint256 numFeatures = features[0].length;
        uint256[][] memory newCentroids = new uint256[][](k);
        totalMovement = 0;
        
        // Initialize new centroids
        for (uint256 i = 0; i < k; i++) {
            newCentroids[i] = new uint256[](numFeatures);
        }
        
        // Accumulate feature values untuk setiap cluster
        for (uint256 i = 0; i < features.length; i++) {
            uint256 cluster = assignments[i];
            for (uint256 j = 0; j < numFeatures; j++) {
                newCentroids[cluster][j] += features[i][j];
            }
        }
        
        // Calculate means dan hitung pergerakan
        for (uint256 i = 0; i < k; i++) {
            if (clusterSizes[i] > 0) {
                for (uint256 j = 0; j < numFeatures; j++) {
                    uint256 newValue = newCentroids[i][j] / clusterSizes[i];
                    uint256 movement;
                    if (newValue >= centroids[i][j]) {
                        movement = newValue - centroids[i][j];
                    } else {
                        movement = centroids[i][j] - newValue;
                    }
                    totalMovement += movement;
                    centroids[i][j] = newValue;
                }
            } else {
                // Handle empty clusters dengan random reinitialization
                uint256 randomIndex = uint256(keccak256(abi.encodePacked(block.timestamp, i))) % features.length;
                for (uint256 j = 0; j < numFeatures; j++) {
                    uint256 movement;
                    if (features[randomIndex][j] >= centroids[i][j]) {
                        movement = features[randomIndex][j] - centroids[i][j];
                    } else {
                        movement = centroids[i][j] - features[randomIndex][j];
                    }
                    totalMovement += movement;
                    centroids[i][j] = features[randomIndex][j];
                }
            }
        }
        
        return totalMovement;
    }
    
    /**
     * @dev Hitung Within-Cluster Sum of Squares (WCSS)
     * @param features Data training
     * @param assignments Cluster assignments
     * @return wcss Nilai WCSS
     */
    function _calculateWCSS(
        uint256[][] calldata features,
        uint256[] memory assignments
    ) internal view returns (uint256 wcss) {
        wcss = 0;
        
        for (uint256 i = 0; i < features.length; i++) {
            uint256 cluster = assignments[i];
            uint256 distance = _euclideanDistance(features[i], centroids[cluster]);
            wcss += (distance * distance) / PRECISION; // squared distance
        }
        
        return wcss;
    }
    
    /**
     * @dev Hitung Silhouette Score
     * @param features Data training
     * @param assignments Cluster assignments
     * @return score Nilai Silhouette Score
     */
    function _calculateSilhouetteScore(
        uint256[][] calldata features,
        uint256[] memory assignments
    ) internal view returns (uint256 score) {
        uint256 numSamples = features.length;
        if (numSamples == 0 || k <= 1) return 0;
        
        uint256 totalScore = 0;
        uint256 validSamples = 0;
        
        for (uint256 i = 0; i < numSamples; i++) {
            uint256 ownCluster = assignments[i];
            
            // Hitung average distance ke samples dalam cluster sendiri (a_i)
            uint256 a_i = _averageDistanceToCluster(features, assignments, i, ownCluster);
            
            // Hitung average distance ke samples dalam cluster terdekat (b_i)
            uint256 b_i = type(uint256).max;
            for (uint256 j = 0; j < k; j++) {
                if (j != ownCluster) {
                    uint256 avgDist = _averageDistanceToCluster(features, assignments, i, j);
                    if (avgDist < b_i) {
                        b_i = avgDist;
                    }
                }
            }
            
            // Hitung silhouette untuk sample i
            if (a_i < b_i) {
                uint256 s_i = (PRECISION * (b_i - a_i)) / (b_i > a_i ? b_i : PRECISION);
                totalScore += s_i;
                validSamples++;
            } else if (a_i > b_i) {
                uint256 s_i = (PRECISION * (b_i - a_i)) / (a_i > b_i ? a_i : PRECISION);
                totalScore += s_i;
                validSamples++;
            } else {
                // a_i == b_i, silhouette = 0
                validSamples++;
            }
        }
        
        return validSamples > 0 ? totalScore / validSamples : 0;
    }
    
    /**
     * @dev Hitung Davies-Bouldin Index
     * @param features Data training
     * @param assignments Cluster assignments
     * @return dbIndex Nilai Davies-Bouldin Index
     */
    function _calculateDaviesBouldinIndex(
        uint256[][] calldata features,
        uint256[] memory assignments
    ) internal view returns (uint256 dbIndex) {
        if (k <= 1) return 0;
        
        // Hitung intra-cluster distances
        uint256[] memory intraClusterDistances = new uint256[](k);
        for (uint256 i = 0; i < k; i++) {
            intraClusterDistances[i] = _averageIntraClusterDistance(features, assignments, i);
        }
        
        // Hitung inter-cluster distances
        uint256[][] memory interClusterDistances = new uint256[][](k);
        for (uint256 i = 0; i < k; i++) {
            interClusterDistances[i] = new uint256[](k);
            for (uint256 j = 0; j < k; j++) {
                if (i != j) {
                    interClusterDistances[i][j] = _distanceBetweenCentroids(i, j);
                }
            }
        }
        
        // Hitung Davies-Bouldin Index
        uint256 totalMaxRatio = 0;
        uint256 validClusters = 0;
        
        for (uint256 i = 0; i < k; i++) {
            uint256 maxRatio = 0;
            for (uint256 j = 0; j < k; j++) {
                if (i != j && interClusterDistances[i][j] > 0) {
                    uint256 ratio = (intraClusterDistances[i] + intraClusterDistances[j]) * PRECISION / interClusterDistances[i][j];
                    if (ratio > maxRatio) {
                        maxRatio = ratio;
                    }
                }
            }
            if (maxRatio > 0) {
                totalMaxRatio += maxRatio;
                validClusters++;
            }
        }
        
        return validClusters > 0 ? totalMaxRatio / validClusters : type(uint256).max;
    }
    
    /**
     * @dev Hitung Euclidean distance antara dua vektor
     * @param a Vektor pertama
     * @param b Vektor kedua
     * @return distance Euclidean distance
     */
    function _euclideanDistance(
        uint256[] memory a,
        uint256[] memory b
    ) internal pure returns (uint256 distance) {
        require(a.length == b.length, "KMeans: vectors must have same length");
        
        uint256 sumSquared = 0;
        for (uint256 i = 0; i < a.length; i++) {
            uint256 diff;
            if (a[i] >= b[i]) {
                diff = a[i] - b[i];
            } else {
                diff = b[i] - a[i];
            }
            sumSquared += (diff * diff) / PRECISION;
        }
        
        distance = ArrayUtils.sqrt(sumSquared * PRECISION);
        return distance;
    }
    
    /**
     * @dev Hitung average distance dari sample ke cluster tertentu
     */
    function _averageDistanceToCluster(
        uint256[][] calldata features,
        uint256[] memory assignments,
        uint256 sampleIndex,
        uint256 cluster
    ) internal view returns (uint256 avgDistance) {
        uint256 totalDistance = 0;
        uint256 count = 0;
        
        for (uint256 i = 0; i < features.length; i++) {
            if (assignments[i] == cluster) {
                totalDistance += _euclideanDistance(features[sampleIndex], features[i]);
                count++;
            }
        }
        
        return count > 0 ? totalDistance / count : type(uint256).max;
    }
    
    /**
     * @dev Hitung average intra-cluster distance
     */
    function _averageIntraClusterDistance(
        uint256[][] calldata features,
        uint256[] memory assignments,
        uint256 cluster
    ) internal view returns (uint256 avgDistance) {
        uint256 totalDistance = 0;
        uint256 count = 0;
        
        for (uint256 i = 0; i < features.length; i++) {
            if (assignments[i] == cluster) {
                for (uint256 j = i + 1; j < features.length; j++) {
                    if (assignments[j] == cluster) {
                        totalDistance += _euclideanDistance(features[i], features[j]);
                        count++;
                    }
                }
            }
        }
        
        return count > 0 ? totalDistance / count : 0;
    }
    
    /**
     * @dev Hitung distance antara dua centroids
     */
    function _distanceBetweenCentroids(
        uint256 cluster1,
        uint256 cluster2
    ) internal view returns (uint256 distance) {
        return _euclideanDistance(centroids[cluster1], centroids[cluster2]);
    }
}