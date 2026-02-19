// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ArrayUtils
 * @dev Utility functions untuk operasi array dengan fixed-point arithmetic
 * @notice Library ini menyediakan fungsi utilitas untuk manipulasi array dan operasi vektor
 * @author Rizky Reza
 */
library ArrayUtils {
    // Precision untuk fixed-point arithmetic (18 decimal places)
    uint256 private constant PRECISION = 1e18;
    

    /**
     * @dev Menghitung jumlah elemen dalam array
     * @param array Array input
     * @return sum Jumlah semua elemen dalam array
     */
    function sum(uint256[] memory array) internal pure returns (uint256 sum) {
        require(array.length > 0, "ArrayUtils: array cannot be empty");
        
        sum = 0;
        for (uint256 i = 0; i < array.length; i++) {
            sum += array[i];
        }
        
    }

    /**
     * @dev Menghitung rata-rata elemen dalam array
     * @param array Array input
     * @return average Rata-rata elemen dalam fixed-point
     */
    function mean(uint256[] memory array) internal pure returns (uint256 average) {
        require(array.length > 0, "ArrayUtils: array cannot be empty");
        
        uint256 total = sum(array);
        average = (total * PRECISION) / array.length;
        
    }

    /**
     * @dev Mencari nilai minimum dalam array
     * @param array Array input
     * @return min Nilai minimum dalam array
     */
    function min(uint256[] memory array) internal pure returns (uint256 min) {
        require(array.length > 0, "ArrayUtils: array cannot be empty");
        
        min = array[0];
        for (uint256 i = 1; i < array.length; i++) {
            if (array[i] < min) {
                min = array[i];
            }
        }
        
    }

    /**
     * @dev Mencari nilai maksimum dalam array
     * @param array Array input
     * @return max Nilai maksimum dalam array
     */
    function max(uint256[] memory array) internal pure returns (uint256 max) {
        require(array.length > 0, "ArrayUtils: array cannot be empty");
        
        max = array[0];
        for (uint256 i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
            }
        }
        
    }

    /**
     * @dev Menghitung dot product dari dua vektor
     * @param a Vektor pertama
     * @param b Vektor kedua
     * @return dotProduct Hasil dot product dalam fixed-point
     */
    function dotProduct(
        uint256[] memory a,
        uint256[] memory b
    ) internal pure returns (uint256 dotProduct) {
        require(a.length == b.length, "ArrayUtils: vectors must have same length");
        require(a.length > 0, "ArrayUtils: vectors cannot be empty");
        
        dotProduct = 0;
        for (uint256 i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
        }
        
    }

    /**
     * @dev Menambahkan dua array secara element-wise
     * @param a Array pertama
     * @param b Array kedua
     * @return result Array hasil penambahan
     */
    function add(
        uint256[] memory a,
        uint256[] memory b
    ) internal pure returns (uint256[] memory result) {
        require(a.length == b.length, "ArrayUtils: arrays must have same length");
        require(a.length > 0, "ArrayUtils: arrays cannot be empty");
        
        result = new uint256[](a.length);
        for (uint256 i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        
    }

    /**
     * @dev Mengurangkan dua array secara element-wise
     * @param a Array pertama
     * @param b Array kedua
     * @return result Array hasil pengurangan
     */
    function subtract(
        uint256[] memory a,
        uint256[] memory b
    ) internal pure returns (uint256[] memory result) {
        require(a.length == b.length, "ArrayUtils: arrays must have same length");
        require(a.length > 0, "ArrayUtils: arrays cannot be empty");
        
        result = new uint256[](a.length);
        for (uint256 i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        
    }

    /**
     * @dev Mengalikan dua array secara element-wise
     * @param a Array pertama
     * @param b Array kedua
     * @return result Array hasil perkalian
     */
    function multiply(
        uint256[] memory a,
        uint256[] memory b
    ) internal pure returns (uint256[] memory result) {
        require(a.length == b.length, "ArrayUtils: arrays must have same length");
        require(a.length > 0, "ArrayUtils: arrays cannot be empty");
        
        result = new uint256[](a.length);
        for (uint256 i = 0; i < a.length; i++) {
            result[i] = (a[i] * b[i]) / PRECISION;
        }
        
    }

    /**
     * @dev Membagi dua array secara element-wise
     * @param a Array pembilang
     * @param b Array penyebut
     * @return result Array hasil pembagian
     */
    function divide(
        uint256[] memory a,
        uint256[] memory b
    ) internal pure returns (uint256[] memory result) {
        require(a.length == b.length, "ArrayUtils: arrays must have same length");
        require(a.length > 0, "ArrayUtils: arrays cannot be empty");
        
        result = new uint256[](a.length);
        for (uint256 i = 0; i < a.length; i++) {
            require(b[i] > 0, "ArrayUtils: division by zero");
            result[i] = (a[i] * PRECISION) / b[i];
        }
        
    }

    /**
     * @dev Menggabungkan dua array
     * @param a Array pertama
     * @param b Array kedua
     * @return result Array hasil penggabungan
     */
    function concatenate(
        uint256[] memory a,
        uint256[] memory b
    ) internal pure returns (uint256[] memory result) {
        result = new uint256[](a.length + b.length);
        
        for (uint256 i = 0; i < a.length; i++) {
            result[i] = a[i];
        }
        
        for (uint256 i = 0; i < b.length; i++) {
            result[a.length + i] = b[i];
        }
        
    }

    /**
     * @dev Mengambil slice dari array
     * @param array Array input
     * @param start Index awal (inklusif)
     * @param end Index akhir (eksklusif)
     * @return slice Array hasil slice
     */
    function slice(
        uint256[] memory array,
        uint256 start,
        uint256 end
    ) internal pure returns (uint256[] memory slice) {
        require(start < end, "ArrayUtils: start must be less than end");
        require(end <= array.length, "ArrayUtils: end index out of bounds");
        
        uint256 length = end - start;
        slice = new uint256[](length);
        
        for (uint256 i = 0; i < length; i++) {
            slice[i] = array[start + i];
        }
        
    }

    /**
     * @dev Mengubah bentuk array 1D menjadi 2D
     * @param array Array input 1D
     * @param rows Jumlah baris
     * @param cols Jumlah kolom
     * @return reshaped Array 2D hasil reshape
     */
    function reshape(
        uint256[] memory array,
        uint256 rows,
        uint256 cols
    ) internal pure returns (uint256[][] memory reshaped) {
        require(array.length == rows * cols, "ArrayUtils: invalid dimensions for reshape");
        
        reshaped = new uint256[][](rows);
        for (uint256 i = 0; i < rows; i++) {
            reshaped[i] = new uint256[](cols);
            for (uint256 j = 0; j < cols; j++) {
                reshaped[i][j] = array[i * cols + j];
            }
        }
        
    }

    /**
     * @dev Mengurutkan array secara ascending
     * @param array Array input
     * @return sorted Array yang sudah diurutkan
     */
    function sort(uint256[] memory array) internal pure returns (uint256[] memory sorted) {
        sorted = new uint256[](array.length);
        
        // Copy array
        for (uint256 i = 0; i < array.length; i++) {
            sorted[i] = array[i];
        }
        
        // Bubble sort (sederhana untuk implementasi on-chain)
        for (uint256 i = 0; i < sorted.length - 1; i++) {
            for (uint256 j = 0; j < sorted.length - i - 1; j++) {
                if (sorted[j] > sorted[j + 1]) {
                    // Swap elements
                    (sorted[j], sorted[j + 1]) = (sorted[j + 1], sorted[j]);
                }
            }
        }
        
    }

    /**
     * @dev Membalik urutan array
     * @param array Array input
     * @return reversed Array yang sudah dibalik
     */
    function reverse(uint256[] memory array) internal pure returns (uint256[] memory reversed) {
        reversed = new uint256[](array.length);
        
        for (uint256 i = 0; i < array.length; i++) {
            reversed[i] = array[array.length - 1 - i];
        }
        
    }

    /**
     * @dev Menghitung variance dari array
     * @param array Array input
     * @return variance Nilai variance dalam fixed-point
     */
    function variance(uint256[] memory array) internal pure returns (uint256 variance) {
        require(array.length > 1, "ArrayUtils: array must have at least 2 elements");
        
        uint256 avg = mean(array);
        uint256 sumSquaredDiff = 0;
        
        for (uint256 i = 0; i < array.length; i++) {
            uint256 diff = array[i] > avg ? array[i] - avg : avg - array[i];
            sumSquaredDiff += (diff * diff) / PRECISION;
        }
        
        variance = (sumSquaredDiff * PRECISION) / (array.length - 1);
        
    }

    /**
     * @dev Menghitung standard deviation dari array
     * @param array Array input
     * @return stdDev Nilai standard deviation dalam fixed-point
     */
    function standardDeviation(uint256[] memory array) internal pure returns (uint256 stdDev) {
        uint256 varValue = variance(array);
        // Simple approximation for square root (Newton's method bisa ditambahkan nanti)
        stdDev = sqrt(varValue);
        
    }

    /**
     * @dev Fungsi helper untuk menghitung square root (approximation)
     * @param x Nilai input
     * @return y Nilai square root approximation
     */
    function sqrt(uint256 x) internal pure returns (uint256 y) {
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
     * @dev Memvalidasi array untuk operasi matematika
     * @param array Array yang akan divalidasi
     * @return isValid True jika array valid
     * @return errorMessage Pesan error jika tidak valid
     */
    function validateArray(uint256[] memory array) internal pure returns (bool isValid, string memory errorMessage) {
        if (array.length == 0) {
            return (false, "ArrayUtils: array cannot be empty");
        }
        
        // Cek untuk overflow potential
        for (uint256 i = 0; i < array.length; i++) {
            if (array[i] > type(uint256).max / PRECISION) {
                return (false, "ArrayUtils: value too large for fixed-point operations");
            }
        }
        
        return (true, "");
    }
}