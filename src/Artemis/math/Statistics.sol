
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./ArrayUtils.sol";

/**
 * @title Statistics
 * @dev Library untuk fungsi statistik dan distribusi dengan fixed-point arithmetic
 * @notice Library ini menyediakan fungsi statistik seperti mean, variance, dan distribusi probability
 * @author Rizky Reza
 */
library Statistics {
    // Precision untuk fixed-point arithmetic (18 decimal places)
    uint256 private constant PRECISION = 1e18;
    uint256 private constant PI = 3141592653589793238; // π dengan 18 decimals
    uint256 private constant E = 2718281828459045235;  // e dengan 18 decimals
    uint256 private constant SQRT_2PI = 2506628274631000502; // √(2π) dengan 18 decimals

    /**
     * @dev Menghitung median dari array
     * @param array Array input
     * @return median Nilai median dalam fixed-point
     */
    function median(uint256[] memory array) internal pure returns (uint256 median) {
        require(array.length > 0, "Statistics: array cannot be empty");
        
        uint256[] memory sorted = ArrayUtils.sort(array);
        
        if (sorted.length % 2 == 0) {
            // Jumlah elemen genap: median = rata-rata dua nilai tengah
            uint256 mid1 = sorted[sorted.length / 2 - 1];
            uint256 mid2 = sorted[sorted.length / 2];
            median = (mid1 + mid2) / 2;
        } else {
            // Jumlah elemen ganjil: median = nilai tengah
            median = sorted[sorted.length / 2];
        }
    }

    /**
     * @dev Menghitung mode dari array (nilai yang paling sering muncul)
     * @param array Array input
     * @return mode Nilai mode
     * @return frequency Frekuensi kemunculan mode
     */
    function mode(uint256[] memory array) internal pure returns (uint256 mode, uint256 frequency) {
        require(array.length > 0, "Statistics: array cannot be empty");
        
        uint256[] memory sorted = ArrayUtils.sort(array);
        
        mode = sorted[0];
        frequency = 1;
        uint256 currentValue = sorted[0];
        uint256 currentFrequency = 1;
        
        for (uint256 i = 1; i < sorted.length; i++) {
            if (sorted[i] == currentValue) {
                currentFrequency++;
            } else {
                if (currentFrequency > frequency) {
                    frequency = currentFrequency;
                    mode = currentValue;
                }
                currentValue = sorted[i];
                currentFrequency = 1;
            }
        }
        
        // Cek untuk elemen terakhir
        if (currentFrequency > frequency) {
            frequency = currentFrequency;
            mode = currentValue;
        }
    }

    /**
     * @dev Menghitung skewness (kemencengan) distribusi
     * @param array Array input
     * @return skewness Nilai skewness dalam fixed-point
     */
    function skewness(uint256[] memory array) internal pure returns (int256 skewness) {
        require(array.length > 2, "Statistics: array must have at least 3 elements");
        
        uint256 meanValue = ArrayUtils.mean(array);
        uint256 stdDev = ArrayUtils.standardDeviation(array);
        
        if (stdDev == 0) {
            return 0; // Tidak ada variasi, skewness = 0
        }
        
        uint256 sumCubedDiff = 0;
        for (uint256 i = 0; i < array.length; i++) {
            int256 diff = int256(array[i]) - int256(meanValue);
            int256 cubedDiff = (diff * diff * diff) / int256(PRECISION * PRECISION);
            sumCubedDiff += uint256(cubedDiff);
        }
        
        uint256 n = array.length;
        skewness = (int256(sumCubedDiff) * int256(PRECISION)) / (int256(n) * int256(stdDev) * int256(stdDev) * int256(stdDev) / int256(PRECISION));
    }

    /**
     * @dev Menghitung kurtosis (keruncingan) distribusi
     * @param array Array input
     * @return kurtosis Nilai kurtosis dalam fixed-point
     */
    function kurtosis(uint256[] memory array) internal pure returns (int256 kurtosis) {
        require(array.length > 3, "Statistics: array must have at least 4 elements");
        
        uint256 meanValue = ArrayUtils.mean(array);
        uint256 varianceValue = ArrayUtils.variance(array);
        
        if (varianceValue == 0) {
            return 0; // Tidak ada variasi, kurtosis = 0
        }
        
        uint256 sumFourthDiff = 0;
        for (uint256 i = 0; i < array.length; i++) {
            int256 diff = int256(array[i]) - int256(meanValue);
            int256 fourthDiff = (diff * diff * diff * diff) / int256(PRECISION * PRECISION * PRECISION);
            sumFourthDiff += uint256(fourthDiff);
        }
        
        uint256 n = array.length;
        kurtosis = (int256(sumFourthDiff) * int256(PRECISION)) / (int256(n) * int256(varianceValue) * int256(varianceValue) / int256(PRECISION)) - 3 * int256(PRECISION);
    }

    /**
     * @dev Menghitung covariance antara dua array
     * @param x Array pertama
     * @param y Array kedua
     * @return covariance Nilai covariance dalam fixed-point
     */
    function covariance(
        uint256[] memory x,
        uint256[] memory y
    ) internal pure returns (int256 covariance) {
        require(x.length == y.length, "Statistics: arrays must have same length");
        require(x.length > 1, "Statistics: arrays must have at least 2 elements");
        
        uint256 meanX = ArrayUtils.mean(x);
        uint256 meanY = ArrayUtils.mean(y);
        
        int256 sumProduct = 0;
        for (uint256 i = 0; i < x.length; i++) {
            int256 diffX = int256(x[i]) - int256(meanX);
            int256 diffY = int256(y[i]) - int256(meanY);
            sumProduct += (diffX * diffY) / int256(PRECISION);
        }
        
        covariance = (sumProduct * int256(PRECISION)) / int256(x.length - 1);
    }

    /**
     * @dev Menghitung correlation coefficient (Pearson) antara dua array
     * @param x Array pertama
     * @param y Array kedua
     * @return correlation Nilai correlation coefficient dalam fixed-point
     */
    function correlation(
        uint256[] memory x,
        uint256[] memory y
    ) internal pure returns (int256 correlation) {
        require(x.length == y.length, "Statistics: arrays must have same length");
        require(x.length > 1, "Statistics: arrays must have at least 2 elements");
        
        int256 cov = covariance(x, y);
        uint256 stdDevX = ArrayUtils.standardDeviation(x);
        uint256 stdDevY = ArrayUtils.standardDeviation(y);
        
        if (stdDevX == 0 || stdDevY == 0) {
            return 0; // Tidak ada variasi, correlation = 0
        }
        
        correlation = (cov * int256(PRECISION)) / (int256(stdDevX) * int256(stdDevY) / int256(PRECISION));
    }

    /**
     * @dev Menghitung probability density function (PDF) untuk distribusi normal
     * @param x Nilai input
     * @param mean Nilai mean distribusi
     * @param stdDev Nilai standard deviation
     * @return pdf Nilai PDF dalam fixed-point
     */
    function normalPDF(
        uint256 x,
        uint256 mean,
        uint256 stdDev
    ) internal pure returns (uint256 pdf) {
        require(stdDev > 0, "Statistics: standard deviation must be positive");
        
        int256 z = (int256(x) - int256(mean)) * int256(PRECISION) / int256(stdDev);
        int256 zSquared = (z * z) / int256(PRECISION);
        
        // PDF(x) = (1 / (σ√(2π))) * e^(-(x-μ)²/(2σ²))
        int256 exponent = -zSquared / (2 * int256(PRECISION));
        uint256 expTerm = exp(uint256(exponent));
        
        pdf = (PRECISION * expTerm) / (stdDev * SQRT_2PI / PRECISION);
    }

    /**
     * @dev Menghitung cumulative distribution function (CDF) untuk distribusi normal (approximation)
     * @param x Nilai input
     * @param mean Nilai mean distribusi
     * @param stdDev Nilai standard deviation
     * @return cdf Nilai CDF dalam fixed-point
     */
    function normalCDF(
        uint256 x,
        uint256 mean,
        uint256 stdDev
    ) internal pure returns (uint256 cdf) {
        require(stdDev > 0, "Statistics: standard deviation must be positive");
        
        int256 z = (int256(x) - int256(mean)) * int256(PRECISION) / int256(stdDev);
        
        // Approximation menggunakan fungsi error (erf)
        uint256 absZ = z < 0 ? uint256(-z) : uint256(z);
        uint256 t = PRECISION / (PRECISION + (absZ * 2316419) / PRECISION);
        
        uint256 poly = t * (t * (t * (t * (t * 133027 + 321409) + 781478) + 3565638) + 31938153) / PRECISION;
        uint256 exponent = (absZ * absZ) / (2 * PRECISION);
        uint256 erf = PRECISION - (poly * exp(exponent)) / PRECISION;
        
        if (z < 0) {
            cdf = (PRECISION - erf) / 2;
        } else {
            cdf = (PRECISION + erf) / 2;
        }
    }

    /**
     * @dev Menghitung quantile untuk distribusi normal (inverse CDF approximation)
     * @param p Probability (0-1 dalam fixed-point)
     * @param mean Nilai mean distribusi
     * @param stdDev Nilai standard deviation
     * @return quantile Nilai quantile
     */
    function normalQuantile(
        uint256 p,
        uint256 mean,
        uint256 stdDev
    ) internal pure returns (uint256 quantile) {
        require(p > 0 && p < PRECISION, "Statistics: probability must be between 0 and 1");
        require(stdDev > 0, "Statistics: standard deviation must be positive");
        
        // Approximation menggunakan metode Beasley-Springer-Moro
        uint256 q = p - PRECISION / 2;
        uint256 r;
        
        if (uint256(abs(int256(q))) <= PRECISION * 42 / 100) { // 0.42
            r = q * q;
            int256 numerator = int256(q) * (((((-25 * 10**15) * int256(r) + 41 * 10**16) * int256(r) - 240 * 10**16) * int256(r) + 89 * 10**17) * int256(r) + 127 * 10**18) / int256(PRECISION);
            int256 denominator = (((((int256(r) + 66 * 10**16) * int256(r) + 112 * 10**17) * int256(r) + 162 * 10**17) * int256(r) + 130 * 10**17) * int256(r) + 84 * 10**17) / int256(PRECISION);
            quantile = uint256(numerator / denominator);
        } else {
            if (q > 0) {
                r = PRECISION - p;
            } else {
                r = p;
            }
            
            r = mySqrt(uint256(-2 * int256(ln(r))));
            quantile = (((((r * 7 + 371 * 10**15) * r + 348 * 10**16) * r + 160 * 10**16) * r + 305 * 10**16) * r + 138 * 10**17) / PRECISION;
            quantile = quantile / (((((r * 5 + 322 * 10**16) * r + 417 * 10**16) * r + 212 * 10**17) * r + 539 * 10**17) * r + 143 * 10**18) / PRECISION;
            
            if (q < 0) {
                quantile = uint256(-int256(quantile));
            }
        }
        
        quantile = mean + (quantile * stdDev) / PRECISION;
    }

    /**
     * @dev Menghitung probability density function (PDF) untuk distribusi uniform
     * @param x Nilai input
     * @param a Batas bawah
     * @param b Batas atas
     * @return pdf Nilai PDF dalam fixed-point
     */
    function uniformPDF(
        uint256 x,
        uint256 a,
        uint256 b
    ) internal pure returns (uint256 pdf) {
        require(b > a, "Statistics: b must be greater than a");
        
        if (x >= a && x <= b) {
            pdf = PRECISION / (b - a);
        } else {
            pdf = 0;
        }
    }

    /**
     * @dev Menghitung cumulative distribution function (CDF) untuk distribusi uniform
     * @param x Nilai input
     * @param a Batas bawah
     * @param b Batas atas
     * @return cdf Nilai CDF dalam fixed-point
     */
    function uniformCDF(
        uint256 x,
        uint256 a,
        uint256 b
    ) internal pure returns (uint256 cdf) {
        require(b > a, "Statistics: b must be greater than a");
        
        if (x < a) {
            cdf = 0;
        } else if (x > b) {
            cdf = PRECISION;
        } else {
            cdf = ((x - a) * PRECISION) / (b - a);
        }
    }

    /**
     * @dev Menghitung z-score untuk nilai tertentu
     * @param x Nilai input
     * @param mean Nilai mean
     * @param stdDev Nilai standard deviation
     * @return zScore Nilai z-score dalam fixed-point
     */
    function zScore(
        uint256 x,
        uint256 mean,
        uint256 stdDev
    ) internal pure returns (int256 zScore) {
        require(stdDev > 0, "Statistics: standard deviation must be positive");
        
        zScore = (int256(x) - int256(mean)) * int256(PRECISION) / int256(stdDev);
    }

    /**
     * @dev Menghitung confidence interval untuk mean
     * @param array Array input
     * @param confidenceLevel Tingkat confidence (0-1 dalam fixed-point)
     * @return lowerBound Batas bawah interval
     * @return upperBound Batas atas interval
     */
    function confidenceInterval(
        uint256[] memory array,
        uint256 confidenceLevel
    ) internal pure returns (uint256 lowerBound, uint256 upperBound) {
        require(array.length > 1, "Statistics: array must have at least 2 elements");
        require(confidenceLevel > 0 && confidenceLevel < PRECISION, "Statistics: confidence level must be between 0 and 1");
        
        uint256 meanValue = ArrayUtils.mean(array);
        uint256 stdDev = ArrayUtils.standardDeviation(array);
        uint256 n = array.length;
        
        // z-value untuk confidence level 95% (approximation)
        uint256 zValue = PRECISION * 196 / 100; // 1.96
        
        uint256 marginOfError = (zValue * stdDev) / (mySqrt(n) * PRECISION);
        
        lowerBound = meanValue - marginOfError;
        upperBound = meanValue + marginOfError;
    }

    /**
     * @dev Fungsi helper untuk menghitung exponential
     * @param x Nilai input
     * @return result Nilai e^x dalam fixed-point
     */
    function exp(uint256 x) internal pure returns (uint256 result) {
        // Approximation menggunakan Taylor series
        result = PRECISION;
        uint256 term = PRECISION;
        
        for (uint256 i = 1; i <= 10; i++) {
            term = (term * x) / (i * PRECISION);
            result += term;
        }
    }

    /**
     * @dev Fungsi helper untuk menghitung natural logarithm
     * @param x Nilai input
     * @return result Nilai ln(x) dalam fixed-point
     */
    function ln(uint256 x) internal pure returns (uint256 result) {
        require(x > 0, "Statistics: ln of non-positive number");
        
        // Approximation menggunakan metode Newton
        if (x == PRECISION) {
            return 0;
        }
        
        uint256 a = x;
        uint256 sum = 0;
        
        while (a >= PRECISION * 2) {
            a = a / 2;
            sum += 693147180559945309; // ln(2) dengan 18 decimals
        }
        
        while (a < PRECISION) {
            a = a * 2;
            sum -= 693147180559945309; // ln(2) dengan 18 decimals
        }
        
        // Newton's method untuk precision yang lebih tinggi
        uint256 z = (a - PRECISION) * PRECISION / (a + PRECISION);
        uint256 z2 = (z * z) / PRECISION;
        result = 2 * z;
        uint256 term = 2 * z * z2 / 3;
        
        for (uint256 i = 3; i <= 15; i += 2) {
            result += term;
            term = (term * z2) / PRECISION;
        }
        
        result += sum;
    }

    /**
     * @dev Fungsi helper untuk menghitung square root (approximation)
     * @param x Nilai input
     * @return y Nilai square root approximation
     */
    function mySqrt(uint256 x) internal pure returns (uint256 y) {
        if (x == 0) return 0;
        
        // Babylonian method (Newton's method untuk square root)
        y = x;
        uint256 z = (y + 1) / 2;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
    }

    /**
     * @dev Fungsi helper untuk menghitung absolute value
     * @param x Nilai input
     * @return absolute Nilai absolute
     */
    function abs(int256 x) internal pure returns (uint256 absolute) {
        if (x < 0) {
            absolute = uint256(-x);
        } else {
            absolute = uint256(x);
        }
    }

    /**
     * @dev Memvalidasi data untuk operasi statistik
     * @param array Array yang akan divalidasi
     * @return isValid True jika data valid
     * @return errorMessage Pesan error jika tidak valid
     */
    function validateData(uint256[] memory array) internal pure returns (bool isValid, string memory errorMessage) {
        if (array.length == 0) {
            return (false, "Statistics: array cannot be empty");
        }
        
        // Cek untuk nilai yang terlalu besar
        for (uint256 i = 0; i < array.length; i++) {
            if (array[i] > type(uint256).max / PRECISION) {
                return (false, "Statistics: value too large for fixed-point operations");
            }
        }
        
        return (true, "");
    }
}