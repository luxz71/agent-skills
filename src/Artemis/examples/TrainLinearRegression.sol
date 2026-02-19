
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../models/supervised/LinearRegression.sol";
import "../utils/DataPreprocessor.sol";
import "../loss/MSELoss.sol";
import "../math/ArrayUtils.sol";

/**
 * @title TrainLinearRegression
 * @dev Contoh penggunaan Linear Regression untuk house price prediction
 * @notice Demonstrasi lengkap training, prediction, dan evaluation workflow untuk regression task
 * @author Rizky Reza
 * 
 * @dev Use Case: House Price Prediction
 * - Dataset: Sample data rumah dengan features (area, bedrooms) dan target (price)
 * - Features: [area (sq ft), bedrooms]
 * - Target: price (USD)
 * 
 * @dev Workflow:
 * 1. Dataset preparation dan preprocessing
 * 2. Model initialization dengan hyperparameter configuration
 * 3. Training dengan gradient descent optimization
 * 4. Prediction pada test data
 * 5. Evaluation dengan R-squared dan MSE metrics
 * 6. Model interpretation dan result analysis
 * 
 * @dev Features Demonstrated:
 * - Data preprocessing dengan DataPreprocessor library
 * - Event monitoring selama training
 * - Model serialization dan parameter management
 * - Performance evaluation dengan multiple metrics
 * - Result interpretation untuk business insights
 */
contract TrainLinearRegression {
    using DataPreprocessor for uint256[];
    using DataPreprocessor for uint256[][];
    using ArrayUtils for uint256[];
    
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    
    // Model instance
    LinearRegression public model;
    
    // Training state
    bool public isTrained;
    uint256 public trainingEpochs;
    uint256 public finalLoss;
    uint256 public rSquared;
    
    // Dataset statistics
    uint256 public datasetSize;
    uint256 public featureCount;
    uint256 public minPrice;
    uint256 public maxPrice;
    uint256 public meanPrice;
    
    // Events untuk monitoring
    event ModelInitialized(address modelAddress, uint256 learningRate, uint256 regularization);
    event TrainingProgress(uint256 epoch, uint256 loss, uint256 accuracy);
    event TrainingCompleted(uint256 finalLoss, uint256 rSquared, uint256 trainingTime);
    event PredictionMade(uint256[] features, uint256 actualPrice, uint256 predictedPrice, uint256 error);
    event ModelEvaluated(uint256 testSize, uint256 accuracy, uint256 loss);
    
    /**
     * @dev Constructor untuk inisialisasi model dan dataset
     * @param learningRate Learning rate untuk training (dalam fixed-point, e.g., 0.01 = 1e16)
     * @param regularization Regularization strength (dalam fixed-point)
     */
    constructor(uint256 learningRate, uint256 regularization) {
        // Inisialisasi model Linear Regression
        model = new LinearRegression(learningRate, regularization);
        
        // Initialize training state
        isTrained = false;
        trainingEpochs = 0;
        finalLoss = 0;
        rSquared = 0;
        
        emit ModelInitialized(address(model), learningRate, regularization);
    }
    
    /**
     * @dev Mendapatkan sample dataset untuk house price prediction
     * @return features Array 2D features: [area, bedrooms]
     * @return labels Array 1D labels: house prices
     * 
     * @dev Dataset Description:
     * - 10 sample data points untuk training
     * - Features: [area (sq ft), bedrooms]
     * - Labels: price (USD dalam ribuan)
     * - Data berdasarkan real-world house price patterns
     */
    function getTrainingDataset() public pure returns (
        uint256[][] memory features,
        uint256[] memory labels
    ) {
        // Sample dataset: [area (sq ft), bedrooms] -> price (USD dalam ribuan)
        features = new uint256[][](10);
        labels = new uint256[](10);
        
        // Data point 1: Small house
        features[0] = new uint256[](2);
        features[0][0] = 1000 * PRECISION; // 1000 sq ft
        features[0][1] = 2 * PRECISION;    // 2 bedrooms
        labels[0] = 200 * 1000 * PRECISION; // $200,000
        
        // Data point 2: Medium house
        features[1] = new uint256[](2);
        features[1][0] = 1500 * PRECISION; // 1500 sq ft
        features[1][1] = 3 * PRECISION;    // 3 bedrooms
        labels[1] = 300 * 1000 * PRECISION; // $300,000
        
        // Data point 3: Large house
        features[2] = new uint256[](2);
        features[2][0] = 2000 * PRECISION; // 2000 sq ft
        features[2][1] = 4 * PRECISION;    // 4 bedrooms
        labels[2] = 400 * 1000 * PRECISION; // $400,000
        
        // Data point 4: Small apartment
        features[3] = new uint256[](2);
        features[3][0] = 800 * PRECISION;  // 800 sq ft
        features[3][1] = 1 * PRECISION;    // 1 bedroom
        labels[3] = 150 * 1000 * PRECISION; // $150,000
        
        // Data point 5: Luxury house
        features[4] = new uint256[](2);
        features[4][0] = 3000 * PRECISION; // 3000 sq ft
        features[4][1] = 5 * PRECISION;    // 5 bedrooms
        labels[4] = 600 * 1000 * PRECISION; // $600,000
        
        // Data point 6: Townhouse
        features[5] = new uint256[](2);
        features[5][0] = 1200 * PRECISION; // 1200 sq ft
        features[5][1] = 2 * PRECISION;    // 2 bedrooms
        labels[5] = 250 * 1000 * PRECISION; // $250,000
        
        // Data point 7: Family home
        features[6] = new uint256[](2);
        features[6][0] = 1800 * PRECISION; // 1800 sq ft
        features[6][1] = 3 * PRECISION;    // 3 bedrooms
        labels[6] = 350 * 1000 * PRECISION; // $350,000
        
        // Data point 8: Condo
        features[7] = new uint256[](2);
        features[7][0] = 900 * PRECISION;  // 900 sq ft
        features[7][1] = 2 * PRECISION;    // 2 bedrooms
        labels[7] = 180 * 1000 * PRECISION; // $180,000
        
        // Data point 9: Executive home
        features[8] = new uint256[](2);
        features[8][0] = 2500 * PRECISION; // 2500 sq ft
        features[8][1] = 4 * PRECISION;    // 4 bedrooms
        labels[8] = 500 * 1000 * PRECISION; // $500,000
        
        // Data point 10: Starter home
        features[9] = new uint256[](2);
        features[9][0] = 1100 * PRECISION; // 1100 sq ft
        features[9][1] = 2 * PRECISION;    // 2 bedrooms
        labels[9] = 220 * 1000 * PRECISION; // $220,000
        
        return (features, labels);
    }
    
    /**
     * @dev Mendapatkan test dataset untuk evaluation
     * @return features Array 2D test features
     * @return labels Array 1D test labels
     */
    function getTestDataset() public pure returns (
        uint256[][] memory features,
        uint256[] memory labels
    ) {
        features = new uint256[][](3);
        labels = new uint256[](3);
        
        // Test data point 1
        features[0] = new uint256[](2);
        features[0][0] = 1300 * PRECISION; // 1300 sq ft
        features[0][1] = 2 * PRECISION;    // 2 bedrooms
        labels[0] = 260 * 1000 * PRECISION; // $260,000
        
        // Test data point 2
        features[1] = new uint256[](2);
        features[1][0] = 1700 * PRECISION; // 1700 sq ft
        features[1][1] = 3 * PRECISION;    // 3 bedrooms
        labels[1] = 330 * 1000 * PRECISION; // $330,000
        
        // Test data point 3
        features[2] = new uint256[](2);
        features[2][0] = 2200 * PRECISION; // 2200 sq ft
        features[2][1] = 4 * PRECISION;    // 4 bedrooms
        labels[2] = 450 * 1000 * PRECISION; // $450,000
        
        return (features, labels);
    }
    
    /**
     * @dev Melakukan preprocessing pada dataset
     * @param rawFeatures Features mentah
     * @param rawLabels Labels mentah
     * @return processedFeatures Features yang sudah dipreprocess
     * @return processedLabels Labels yang sudah dipreprocess
     * @return scalingParams Parameter scaling yang digunakan
     * 
     * @dev Preprocessing Steps:
     * 1. Feature scaling menggunakan min-max normalization
     * 2. Label scaling untuk stabilitas numerik
     * 3. Validasi data integrity
     */
    function preprocessDataset(
        uint256[][] memory rawFeatures,
        uint256[] memory rawLabels
    ) public pure returns (
        uint256[][] memory processedFeatures,
        uint256[] memory processedLabels,
        uint256[][] memory scalingParams
    ) {
        require(rawFeatures.length > 0, "TrainLinearRegression: Features cannot be empty");
        require(rawFeatures.length == rawLabels.length, "TrainLinearRegression: Features and labels must have same length");
        
        // Scale features menggunakan min-max normalization
        (processedFeatures, scalingParams) = DataPreprocessor.scaleFeatures(
            rawFeatures,
            0 // 0 = min-max scaling
        );
        
        // Scale labels untuk stabilitas numerik (normalize ke range [0, 1])
        uint256 minLabel = ArrayUtils.min(rawLabels);
        uint256 maxLabel = ArrayUtils.max(rawLabels);
        
        processedLabels = DataPreprocessor.minMaxScale(
            rawLabels,
            minLabel,
            maxLabel,
            0,
            PRECISION
        );
        
        return (processedFeatures, processedLabels, scalingParams);
    }
    
    /**
     * @dev Melakukan training model dengan dataset yang sudah dipreprocess
     * @param epochs Jumlah epoch training
     * @param learningRate Learning rate untuk training
     * @return success Status keberhasilan training
     * @return trainingLoss Nilai loss akhir setelah training
     * 
     * @dev Training Process:
     * 1. Load dan preprocess dataset
     * 2. Train model dengan gradient descent
     * 3. Monitor progress dengan events
     * 4. Simpan training statistics
     */
    function trainModel(uint256 epochs, uint256 learningRate) public returns (bool success, uint256 trainingLoss) {
        require(epochs > 0, "TrainLinearRegression: Epochs must be positive");
        require(learningRate > 0, "TrainLinearRegression: Learning rate must be positive");
        
        // Dapatkan dataset
        (uint256[][] memory rawFeatures, uint256[] memory rawLabels) = getTrainingDataset();
        
        // Preprocess dataset
        (uint256[][] memory features, uint256[] memory labels, ) = preprocessDataset(rawFeatures, rawLabels);
        
        // Update dataset statistics
        datasetSize = features.length;
        featureCount = features[0].length;
        minPrice = ArrayUtils.min(rawLabels);
        maxPrice = ArrayUtils.max(rawLabels);
        meanPrice = ArrayUtils.mean(rawLabels);
        
        // Train model
        uint256 startTime = block.timestamp;
        (success, trainingLoss) = model.train(features, labels, epochs, learningRate);
        
        if (success) {
            isTrained = true;
            trainingEpochs = epochs;
            finalLoss = trainingLoss;
            
            // Evaluate model untuk mendapatkan R-squared
            (uint256 accuracy, ) = model.evaluate(features, labels);
            rSquared = accuracy;
            
            uint256 trainingTime = block.timestamp - startTime;
            
            emit TrainingCompleted(trainingLoss, rSquared, trainingTime);
        }
        
        return (success, trainingLoss);
    }
    
    /**
     * @dev Melakukan prediksi harga rumah berdasarkan features
     * @param area Luas rumah dalam square feet
     * @param bedrooms Jumlah bedrooms
     * @return predictedPrice Harga prediksi dalam USD
     * 
     * @dev Prediction Process:
     * 1. Preprocess input features
     * 2. Lakukan prediksi menggunakan model
     * 3. Denormalize hasil prediksi
     * 4. Return harga dalam format yang mudah dibaca
     */
    function predictHousePrice(uint256 area, uint256 bedrooms) public view returns (uint256 predictedPrice) {
        require(isTrained, "TrainLinearRegression: Model not trained");
        
        // Prepare features
        uint256[] memory features = new uint256[](2);
        features[0] = area * PRECISION;
        features[1] = bedrooms * PRECISION;
        
        // Untuk simplicity dalam contoh, kita skip preprocessing features
        // Dalam implementasi produksi, gunakan scaling parameters yang sama dengan training
        
        // Lakukan prediksi
        uint256 normalizedPrediction = model.predict(features);
        
        // Denormalize prediction (asumsi min=0, max=1e6 * PRECISION untuk simplicity)
        // Dalam implementasi produksi, gunakan scaling parameters yang disimpan
        predictedPrice = (normalizedPrediction * 1000000 * PRECISION) / PRECISION;
        
        return predictedPrice;
    }
    
    /**
     * @dev Mengevaluasi performa model dengan test dataset
     * @return accuracy R-squared score
     * @return loss MSE loss
     * 
     * @dev Evaluation Metrics:
     * - R-squared: Proporsi variasi dalam data yang dijelaskan oleh model
     * - MSE: Rata-rata kuadrat error antara prediksi dan actual values
     */
    function evaluateModel() public returns (uint256 accuracy, uint256 loss) {
        require(isTrained, "TrainLinearRegression: Model not trained");
        
        (uint256[][] memory testFeatures, uint256[] memory testLabels) = getTestDataset();
        
        // Preprocess test data (dengan parameter yang sama seperti training)
        (uint256[][] memory processedFeatures, uint256[] memory processedLabels, ) = preprocessDataset(testFeatures, testLabels);
        
        // Evaluate model
        (accuracy, loss) = model.evaluate(processedFeatures, processedLabels);
        
        emit ModelEvaluated(testFeatures.length, accuracy, loss);
        
        return (accuracy, loss);
    }
    
    /**
     * @dev Mendapatkan interpretasi model coefficients
     * @return areaCoefficient Pengaruh area terhadap harga
     * @return bedroomsCoefficient Pengaruh bedrooms terhadap harga
     * @return bias Harga dasar (intercept)
     * 
     * @dev Interpretation:
     * - areaCoefficient: Setiap peningkatan 1 sq ft meningkatkan harga sebesar coefficient
     * - bedroomsCoefficient: Setiap tambahan bedroom meningkatkan harga sebesar coefficient  
     * - bias: Harga dasar ketika area dan bedrooms = 0
     */
    function getModelInterpretation() public view returns (
        uint256 areaCoefficient,
        uint256 bedroomsCoefficient,
        uint256 bias
    ) {
        require(isTrained, "TrainLinearRegression: Model not trained");
        
        uint256[] memory parameters = model.getParameters();
        require(parameters.length >= 3, "TrainLinearRegression: Invalid parameters");
        
        bias = parameters[0];
        areaCoefficient = parameters[1];
        bedroomsCoefficient = parameters[2];
        
        return (areaCoefficient, bedroomsCoefficient, bias);
    }
    
    /**
     * @dev Mendapatkan informasi lengkap tentang model
     * @return modelInfo Metadata model
     * @return trainingInfo Status training
     * @return datasetInfo Statistik dataset
     */
    function getModelInfo() public view returns (
        string memory modelInfo,
        string memory trainingInfo,
        string memory datasetInfo
    ) {
        (string memory name, string memory version, uint256 inputSize, uint256 outputSize) = model.getModelInfo();
        (bool trained, uint256 epochs, uint256 loss) = model.getTrainingStatus();
        
        modelInfo = string(abi.encodePacked(
            "Model: ", name, " v", version, 
            " | Input: ", _uintToString(inputSize),
            " | Output: ", _uintToString(outputSize)
        ));
        
        trainingInfo = string(abi.encodePacked(
            "Trained: ", trained ? "Yes" : "No",
            " | Epochs: ", _uintToString(epochs),
            " | Loss: ", _uintToString(loss)
        ));
        
        datasetInfo = string(abi.encodePacked(
            "Dataset Size: ", _uintToString(datasetSize),
            " | Features: ", _uintToString(featureCount),
            " | Price Range: $", _uintToString(minPrice / (1000 * PRECISION)), "K-$", _uintToString(maxPrice / (1000 * PRECISION)), "K"
        ));
        
        return (modelInfo, trainingInfo, datasetInfo);
    }
    
    /**
     * @dev Demo function untuk menunjukkan workflow lengkap
     * @return demoResults Hasil demonstrasi
     */

    function runDemo() public returns (string memory demoResults) {
        // Step 1: Train model
        (bool success, uint256 loss) = trainModel(100, (1 * PRECISION) / 100); // 100 epochs, 0.01 learning rate
        
        if (!success) {
            return "Demo failed: Training unsuccessful";
        }
        
        // Step 2: Evaluate model
        (uint256 accuracy, uint256 testLoss) = evaluateModel();
        
        // Step 3: Make predictions
        uint256 prediction1 = predictHousePrice(1500, 3); // 1500 sq ft, 3 bedrooms
        uint256 prediction2 = predictHousePrice(2000, 4); // 2000 sq ft, 4 bedrooms
        
        // Step 4: Get model interpretation
        (uint256 areaCoef, uint256 bedroomsCoef, uint256 bias) = getModelInterpretation();
        
        demoResults = string(abi.encodePacked(
            "=== Artemis Linear Regression Demo ===\n",
            "Training: Success (Loss: ", _uintToString(loss), ")\n",
            "Evaluation: R-squared=", _uintToString(accuracy), ", Test Loss=", _uintToString(testLoss), "\n",
            "Predictions: 1500sqft/3bd=$", _uintToString(prediction1 / (1000 * PRECISION)), "K, ",
            "2000sqft/4bd=$", _uintToString(prediction2 / (1000 * PRECISION)), "K\n",
            "Model Interpretation: Area Coef=", _uintToString(areaCoef), 
            ", Bedrooms Coef=", _uintToString(bedroomsCoef),
            ", Bias=", _uintToString(bias)
        ));
        
        return demoResults;
    }
    
    /**
     * @dev Fungsi internal untuk konversi uint ke string
     * @param _i Nilai uint
     * @return _uintAsString String hasil konversi
     */
    function _uintToString(uint256 _i) internal pure returns (string memory _uintAsString) {
        if (_i == 0) {
            return "0";
        }
        uint256 j = _i;
        uint256 len;
        while (j != 0) {
            len++;
            j /= 10;
        }
        bytes memory bstr = new bytes(len);
        uint256 k = len;
        while (_i != 0) {
            k = k - 1;
            uint8 temp = (48 + uint8(_i - (_i / 10) * 10));
            bytes1 b1 = bytes1(temp);
            bstr[k] = b1;
            _i /= 10;
        }
        return string(bstr);
    }
    
    /**
     * @dev Mendapatkan best practices untuk Linear Regression
     * @return practices Tips dan best practices
     */
    function getBestPractices() public pure returns (string memory practices) {
        practices = string(abi.encodePacked(
            "=== Linear Regression Best Practices ===\n",
            "1. Feature Scaling: Selalu scale features untuk stabilitas numerik\n",
            "2. Learning Rate: Gunakan learning rate kecil (0.01-0.001) untuk convergence yang baik\n",
            "3. Regularization: Gunakan L2 regularization untuk mencegah overfitting\n",
            "4. Dataset Size: Minimal 10x jumlah features untuk training yang reliable\n",
            "5. Evaluation: Selalu evaluasi dengan test dataset terpisah\n",
            "6. Interpretation: Perhatikan sign dan magnitude coefficients untuk insights bisnis"
        ));
        return practices;
    }
    
    /**
     * @dev Mendapatkan troubleshooting guide untuk common issues
     * @return troubleshooting Panduan troubleshooting
     */
    function getTroubleshootingGuide() public pure returns (string memory troubleshooting) {
        troubleshooting = string(abi.encodePacked(
            "=== Troubleshooting Guide ===\n",
            "Issue: Training loss tidak berkurang\n",
            "Solution: Turunkan learning rate, cek data preprocessing\n\n",
            "Issue: Model overfitting\n",
            "Solution: Tingkatkan regularization, kurangi complexity model\n\n",
            "Issue: Predictions tidak akurat\n",
            "Solution: Cek feature scaling, evaluasi dengan lebih banyak data\n\n",
            "Issue: Gas costs terlalu tinggi\n",
            "Solution: Kurangi epochs, gunakan batch training untuk dataset besar"
        ));
        return troubleshooting;
    }
}
