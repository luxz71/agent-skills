// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../math/ArrayUtils.sol";
import "../math/Statistics.sol";

/**
 * @title DataPreprocessor
 * @dev Library untuk preprocessing data machine learning dengan fixed-point arithmetic
 * @notice Library ini menyediakan fungsi untuk normalisasi, scaling, dan feature engineering data
 * @author Rizky Reza
 */
library DataPreprocessor {
    // Precision untuk fixed-point arithmetic (18 decimal places)
    uint256 private constant PRECISION = 1e18;


    /**
     * @dev Melakukan min-max scaling pada array data
     * @param data Array data input
     * @param featureMin Nilai minimum untuk feature
     * @param featureMax Nilai maksimum untuk feature
     * @param targetMin Nilai minimum target (biasanya 0)
     * @param targetMax Nilai maksimum target (biasanya 1)
     * @return scaledData Array data yang sudah discale
     */
    function minMaxScale(
        uint256[] memory data,
        uint256 featureMin,
        uint256 featureMax,
        uint256 targetMin,
        uint256 targetMax
    ) internal pure returns (uint256[] memory scaledData) {
        require(data.length > 0, "DataPreprocessor: data cannot be empty");
        require(featureMax > featureMin, "DataPreprocessor: featureMax must be greater than featureMin");
        require(targetMax > targetMin, "DataPreprocessor: targetMax must be greater than targetMin");
        
        scaledData = new uint256[](data.length);
        uint256 featureRange = featureMax - featureMin;
        uint256 targetRange = targetMax - targetMin;
        
        for (uint256 i = 0; i < data.length; i++) {
            if (featureRange == 0) {
                scaledData[i] = (targetMin + targetMax) / 2; // Jika range 0, gunakan nilai tengah
            } else {
                scaledData[i] = targetMin + ((data[i] - featureMin) * targetRange) / featureRange;
            }
        }
        
    }

    /**
     * @dev Melakukan standardization (z-score normalization) pada array data
     * @param data Array data input
     * @param mean Nilai mean data
     * @param stdDev Nilai standard deviation data
     * @return standardizedData Array data yang sudah distandardisasi
     */
    function standardize(
        uint256[] memory data,
        uint256 mean,
        uint256 stdDev
    ) internal pure returns (uint256[] memory standardizedData) {
        require(data.length > 0, "DataPreprocessor: data cannot be empty");
        require(stdDev > 0, "DataPreprocessor: standard deviation must be positive");
        
        standardizedData = new uint256[](data.length);
        
        for (uint256 i = 0; i < data.length; i++) {
            int256 zScore = (int256(data[i]) - int256(mean)) * int256(PRECISION) / int256(stdDev);
            standardizedData[i] = uint256(zScore);
        }
        
    }

    /**
     * @dev Melakukan standardization dengan menghitung mean dan stdDev dari data
     * @param data Array data input
     * @return standardizedData Array data yang sudah distandardisasi
     * @return computedMean Mean yang dihitung dari data
     * @return computedStdDev Standard deviation yang dihitung dari data
     */
    function standardizeAuto(
        uint256[] memory data
    ) internal pure returns (
        uint256[] memory standardizedData,
        uint256 computedMean,
        uint256 computedStdDev
    ) {
        require(data.length > 1, "DataPreprocessor: data must have at least 2 elements");
        
        computedMean = ArrayUtils.mean(data);
        computedStdDev = ArrayUtils.standardDeviation(data);
        
        if (computedStdDev == 0) {
            // Jika tidak ada variasi, kembalikan array nol
            standardizedData = new uint256[](data.length);
            for (uint256 i = 0; i < data.length; i++) {
                standardizedData[i] = 0;
            }
        } else {
            standardizedData = standardize(data, computedMean, computedStdDev);
        }
        
    }

    /**
     * @dev Melakukan normalization data ke range [0, 1]
     * @param data Array data input
     * @return normalizedData Array data yang sudah dinormalisasi
     */
    function normalize(
        uint256[] memory data
    ) internal pure returns (uint256[] memory normalizedData) {
        require(data.length > 0, "DataPreprocessor: data cannot be empty");
        
        uint256 minVal = ArrayUtils.min(data);
        uint256 maxVal = ArrayUtils.max(data);
        
        normalizedData = minMaxScale(data, minVal, maxVal, 0, PRECISION);
        
    }

    /**
     * @dev Melakukan robust scaling menggunakan median dan IQR
     * @param data Array data input
     * @return scaledData Array data yang sudah discale
     */
    function robustScale(
        uint256[] memory data
    ) internal pure returns (uint256[] memory scaledData) {
        require(data.length > 0, "DataPreprocessor: data cannot be empty");
        
        uint256 medianVal = Statistics.median(data);
        
        // Hitung IQR (Interquartile Range)
        uint256 q1 = quantile(data, PRECISION / 4); // 25th percentile
        uint256 q3 = quantile(data, PRECISION * 3 / 4); // 75th percentile
        uint256 iqr = q3 - q1;
        
        scaledData = new uint256[](data.length);
        
        if (iqr == 0) {
            // Jika IQR = 0, gunakan standardization sebagai fallback
            (scaledData,,) = standardizeAuto(data);
        } else {
            for (uint256 i = 0; i < data.length; i++) {
                int256 robustZ = (int256(data[i]) - int256(medianVal)) * int256(PRECISION) / int256(iqr);
                scaledData[i] = uint256(robustZ);
            }
        }
        
    }

    /**
     * @dev Menghitung quantile dari data
     * @param data Array data input
     * @param p Probability (0-1 dalam fixed-point)
     * @return quantileValue Nilai quantile
     */
    function quantile(
        uint256[] memory data,
        uint256 p
    ) internal pure returns (uint256 quantileValue) {
        require(data.length > 0, "DataPreprocessor: data cannot be empty");
        require(p >= 0 && p <= PRECISION, "DataPreprocessor: probability must be between 0 and 1");
        
        uint256[] memory sorted = ArrayUtils.sort(data);
        uint256 index = (p * sorted.length) / PRECISION;
        
        if (index >= sorted.length) {
            quantileValue = sorted[sorted.length - 1];
        } else {
            quantileValue = sorted[index];
        }
    }

    /**
     * @dev Melakukan one-hot encoding untuk categorical data
     * @param categories Array kategori input
     * @param uniqueCategories Array kategori unik
     * @return encodedData Array 2D hasil encoding
     */
    function oneHotEncode(
        uint256[] memory categories,
        uint256[] memory uniqueCategories
    ) internal pure returns (uint256[][] memory encodedData) {
        require(categories.length > 0, "DataPreprocessor: categories cannot be empty");
        require(uniqueCategories.length > 0, "DataPreprocessor: uniqueCategories cannot be empty");
        
        encodedData = new uint256[][](categories.length);
        
        for (uint256 i = 0; i < categories.length; i++) {
            encodedData[i] = new uint256[](uniqueCategories.length);
            for (uint256 j = 0; j < uniqueCategories.length; j++) {
                if (categories[i] == uniqueCategories[j]) {
                    encodedData[i][j] = PRECISION; // 1.0 dalam fixed-point
                } else {
                    encodedData[i][j] = 0;
                }
            }
        }
        
    }

    /**
     * @dev Melakukan feature scaling untuk matriks (per kolom)
     * @param features Matriks features (baris x kolom)
     * @param scalingType Tipe scaling (0: min-max, 1: standardization, 2: normalization)
     * @return scaledFeatures Matriks features yang sudah discale
     * @return scalingParams Parameter scaling yang digunakan
     */
    function scaleFeatures(
        uint256[][] memory features,
        uint256 scalingType
    ) internal pure returns (
        uint256[][] memory scaledFeatures,
        uint256[][] memory scalingParams
    ) {
        require(features.length > 0, "DataPreprocessor: features cannot be empty");
        uint256 numFeatures = features[0].length;
        require(numFeatures > 0, "DataPreprocessor: features must have columns");
        
        scaledFeatures = new uint256[][](features.length);
        scalingParams = new uint256[][](2); // [min/mean, max/stdDev] untuk setiap feature
        
        // Inisialisasi scaling parameters
        scalingParams[0] = new uint256[](numFeatures); // min atau mean
        scalingParams[1] = new uint256[](numFeatures); // max atau stdDev
        
        // Hitung parameters untuk setiap feature
        for (uint256 j = 0; j < numFeatures; j++) {
            uint256[] memory featureColumn = extractColumn(features, j);
            
            if (scalingType == 0) { // Min-max scaling
                scalingParams[0][j] = ArrayUtils.min(featureColumn);
                scalingParams[1][j] = ArrayUtils.max(featureColumn);
            } else if (scalingType == 1) { // Standardization
                scalingParams[0][j] = ArrayUtils.mean(featureColumn);
                scalingParams[1][j] = ArrayUtils.standardDeviation(featureColumn);
            } else { // Normalization (default)
                scalingParams[0][j] = ArrayUtils.min(featureColumn);
                scalingParams[1][j] = ArrayUtils.max(featureColumn);
            }
        }
        
        // Apply scaling untuk setiap data point
        for (uint256 i = 0; i < features.length; i++) {
            scaledFeatures[i] = new uint256[](numFeatures);
            for (uint256 j = 0; j < numFeatures; j++) {
                if (scalingType == 0) { // Min-max scaling
                    scaledFeatures[i][j] = minMaxScaleSingle(
                        features[i][j],
                        scalingParams[0][j],
                        scalingParams[1][j],
                        0,
                        PRECISION
                    );
                } else if (scalingType == 1) { // Standardization
                    if (scalingParams[1][j] > 0) {
                        int256 zScore = (int256(features[i][j]) - int256(scalingParams[0][j])) * int256(PRECISION) / int256(scalingParams[1][j]);
                        scaledFeatures[i][j] = uint256(zScore);
                    } else {
                        scaledFeatures[i][j] = 0;
                    }
                } else { // Normalization
                    scaledFeatures[i][j] = minMaxScaleSingle(
                        features[i][j],
                        scalingParams[0][j],
                        scalingParams[1][j],
                        0,
                        PRECISION
                    );
                }
            }
        }
        
    }

    /**
     * @dev Mengekstrak kolom dari matriks
     * @param matrix Matriks input
     * @param columnIndex Index kolom yang akan diekstrak
     * @return column Array kolom yang diekstrak
     */
    function extractColumn(
        uint256[][] memory matrix,
        uint256 columnIndex
    ) internal pure returns (uint256[] memory column) {
        require(matrix.length > 0, "DataPreprocessor: matrix cannot be empty");
        require(columnIndex < matrix[0].length, "DataPreprocessor: column index out of bounds");
        
        column = new uint256[](matrix.length);
        for (uint256 i = 0; i < matrix.length; i++) {
            column[i] = matrix[i][columnIndex];
        }
    }

    /**
     * @dev Min-max scaling untuk single value
     * @param value Nilai input
     * @param featureMin Nilai minimum feature
     * @param featureMax Nilai maksimum feature
     * @param targetMin Nilai minimum target
     * @param targetMax Nilai maksimum target
     * @return scaledValue Nilai yang sudah discale
     */
    function minMaxScaleSingle(
        uint256 value,
        uint256 featureMin,
        uint256 featureMax,
        uint256 targetMin,
        uint256 targetMax
    ) internal pure returns (uint256 scaledValue) {
        if (featureMax == featureMin) {
            scaledValue = (targetMin + targetMax) / 2;
        } else {
            scaledValue = targetMin + ((value - featureMin) * (targetMax - targetMin)) / (featureMax - featureMin);
        }
    }

    /**
     * @dev Menangani missing values dengan berbagai strategi
     * @param data Array data dengan missing values (diwakili oleh nilai khusus)
     * @param strategy Strategi handling (0: mean, 1: median, 2: mode, 3: constant)
     * @param fillValue Nilai untuk mengisi jika strategy = constant
     * @return filledData Array data yang sudah diisi
     */
    function handleMissingValues(
        uint256[] memory data,
        uint256 strategy,
        uint256 fillValue
    ) internal pure returns (uint256[] memory filledData) {
        require(data.length > 0, "DataPreprocessor: data cannot be empty");
        
        filledData = new uint256[](data.length);
        uint256 replacementValue;
        
        // Tentukan replacement value berdasarkan strategy
        if (strategy == 0) { // Mean
            replacementValue = ArrayUtils.mean(data);
        } else if (strategy == 1) { // Median
            replacementValue = Statistics.median(data);
        } else if (strategy == 2) { // Mode
            (replacementValue, ) = Statistics.mode(data);
        } else { // Constant
            replacementValue = fillValue;
        }
        
        // Isi missing values
        for (uint256 i = 0; i < data.length; i++) {
            if (data[i] == type(uint256).max) { // Asumsi: type(uint256).max mewakili missing value
                filledData[i] = replacementValue;
            } else {
                filledData[i] = data[i];
            }
        }
        
    }

    /**
     * @dev Memvalidasi data untuk preprocessing
     * @param data Array data yang akan divalidasi
     * @return isValid True jika data valid
     * @return errorMessage Pesan error jika tidak valid
     */
    function validateData(
        uint256[] memory data
    ) internal pure returns (bool isValid, string memory errorMessage) {
        if (data.length == 0) {
            return (false, "DataPreprocessor: data cannot be empty");
        }
        
        // Cek untuk nilai yang terlalu besar
        for (uint256 i = 0; i < data.length; i++) {
            if (data[i] > type(uint256).max / PRECISION && data[i] != type(uint256).max) {
                return (false, "DataPreprocessor: value too large for fixed-point operations");
            }
        }
        
        return (true, "");
    }

    /**
     * @dev Memvalidasi matriks features untuk preprocessing
     * @param features Matriks features yang akan divalidasi
     * @return isValid True jika features valid
     * @return errorMessage Pesan error jika tidak valid
     */
    function validateFeatures(
        uint256[][] memory features
    ) internal pure returns (bool isValid, string memory errorMessage) {
        if (features.length == 0) {
            return (false, "DataPreprocessor: features cannot be empty");
        }
        
        uint256 numFeatures = features[0].length;
        if (numFeatures == 0) {
            return (false, "DataPreprocessor: features must have columns");
        }
        
        // Cek konsistensi jumlah kolom
        for (uint256 i = 1; i < features.length; i++) {
            if (features[i].length != numFeatures) {
                return (false, "DataPreprocessor: inconsistent number of features");
            }
        }
        
        // Cek untuk nilai yang terlalu besar
        for (uint256 i = 0; i < features.length; i++) {
            for (uint256 j = 0; j < numFeatures; j++) {
                if (features[i][j] > type(uint256).max / PRECISION && features[i][j] != type(uint256).max) {
                    return (false, "DataPreprocessor: value too large for fixed-point operations");
                }
            }
        }
        
        return (true, "");
    }
}