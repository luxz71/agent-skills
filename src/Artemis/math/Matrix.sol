
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Matrix
 * @dev Library untuk operasi matriks dan linear algebra dengan fixed-point arithmetic
 * @notice Library ini menyediakan fungsi untuk operasi matriks seperti perkalian, transpose, dan dekomposisi
 * @author Rizky Reza
 */
library Matrix {
    // Precision untuk fixed-point arithmetic (18 decimal places)
    uint256 private constant PRECISION = 1e18;

    /**
     * @dev Melakukan perkalian matriks A (m x n) dengan matriks B (n x p)
     * @param a Matriks pertama (m x n)
     * @param b Matriks kedua (n x p)
     * @return result Matriks hasil perkalian (m x p)
     */
    function multiply(
        uint256[][] memory a,
        uint256[][] memory b
    ) internal pure returns (uint256[][] memory result) {
        uint256 m = a.length;
        require(m > 0, "Matrix: matrix A cannot be empty");
        uint256 n = a[0].length;
        require(n > 0, "Matrix: matrix A must have columns");
        
        uint256 p = b.length;
        require(p > 0, "Matrix: matrix B cannot be empty");
        require(b[0].length > 0, "Matrix: matrix B must have columns");
        require(n == p, "Matrix: incompatible dimensions for multiplication");
        
        uint256 q = b[0].length;
        
        result = new uint256[][](m);
        for (uint256 i = 0; i < m; i++) {
            result[i] = new uint256[](q);
            for (uint256 j = 0; j < q; j++) {
                uint256 sum = 0;
                for (uint256 k = 0; k < n; k++) {
                    sum += (a[i][k] * b[k][j]) / PRECISION;
                }
                result[i][j] = sum;
            }
        }
    }

    /**
     * @dev Melakukan penambahan dua matriks dengan dimensi yang sama
     * @param a Matriks pertama
     * @param b Matriks kedua
     * @return result Matriks hasil penambahan
     */
    function add(
        uint256[][] memory a,
        uint256[][] memory b
    ) internal pure returns (uint256[][] memory result) {
        uint256 rows = a.length;
        require(rows > 0, "Matrix: matrix A cannot be empty");
        require(rows == b.length, "Matrix: matrices must have same number of rows");
        
        uint256 cols = a[0].length;
        require(cols > 0, "Matrix: matrix A must have columns");
        require(cols == b[0].length, "Matrix: matrices must have same number of columns");
        
        result = new uint256[][](rows);
        for (uint256 i = 0; i < rows; i++) {
            result[i] = new uint256[](cols);
            for (uint256 j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
    }

    /**
     * @dev Melakukan pengurangan dua matriks dengan dimensi yang sama
     * @param a Matriks pertama
     * @param b Matriks kedua
     * @return result Matriks hasil pengurangan
     */
    function subtract(
        uint256[][] memory a,
        uint256[][] memory b
    ) internal pure returns (uint256[][] memory result) {
        uint256 rows = a.length;
        require(rows > 0, "Matrix: matrix A cannot be empty");
        require(rows == b.length, "Matrix: matrices must have same number of rows");
        
        uint256 cols = a[0].length;
        require(cols > 0, "Matrix: matrix A must have columns");
        require(cols == b[0].length, "Matrix: matrices must have same number of columns");
        
        result = new uint256[][](rows);
        for (uint256 i = 0; i < rows; i++) {
            result[i] = new uint256[](cols);
            for (uint256 j = 0; j < cols; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
    }

    /**
     * @dev Melakukan transpose matriks
     * @param matrix Matriks input
     * @return transposed Matriks yang sudah ditranspose
     */
    function transpose(
        uint256[][] memory matrix
    ) internal pure returns (uint256[][] memory transposed) {
        uint256 rows = matrix.length;
        require(rows > 0, "Matrix: matrix cannot be empty");
        uint256 cols = matrix[0].length;
        require(cols > 0, "Matrix: matrix must have columns");
        
        transposed = new uint256[][](cols);
        for (uint256 i = 0; i < cols; i++) {
            transposed[i] = new uint256[](rows);
            for (uint256 j = 0; j < rows; j++) {
                transposed[i][j] = matrix[j][i];
            }
        }
    }

    /**
     * @dev Mengalikan matriks dengan skalar
     * @param matrix Matriks input
     * @param scalar Skalar pengali
     * @return result Matriks hasil perkalian dengan skalar
     */
    function multiplyScalar(
        uint256[][] memory matrix,
        uint256 scalar
    ) internal pure returns (uint256[][] memory result) {
        uint256 rows = matrix.length;
        require(rows > 0, "Matrix: matrix cannot be empty");
        uint256 cols = matrix[0].length;
        require(cols > 0, "Matrix: matrix must have columns");
        
        result = new uint256[][](rows);
        for (uint256 i = 0; i < rows; i++) {
            result[i] = new uint256[](cols);
            for (uint256 j = 0; j < cols; j++) {
                result[i][j] = (matrix[i][j] * scalar) / PRECISION;
            }
        }
    }

    /**
     * @dev Mendapatkan diagonal utama matriks
     * @param matrix Matriks input
     * @return diagonal Array elemen diagonal
     */
    function getDiagonal(
        uint256[][] memory matrix
    ) internal pure returns (uint256[] memory diagonal) {
        uint256 rows = matrix.length;
        require(rows > 0, "Matrix: matrix cannot be empty");
        uint256 cols = matrix[0].length;
        require(cols > 0, "Matrix: matrix must have columns");
        require(rows == cols, "Matrix: matrix must be square for diagonal");
        
        diagonal = new uint256[](rows);
        for (uint256 i = 0; i < rows; i++) {
            diagonal[i] = matrix[i][i];
        }
    }

    /**
     * @dev Membuat matriks identitas dengan ukuran n x n
     * @param n Ukuran matriks
     * @return identity Matriks identitas
     */
    function identity(uint256 n) internal pure returns (uint256[][] memory identity) {
        require(n > 0, "Matrix: size must be positive");
        
        identity = new uint256[][](n);
        for (uint256 i = 0; i < n; i++) {
            identity[i] = new uint256[](n);
            for (uint256 j = 0; j < n; j++) {
                identity[i][j] = (i == j) ? PRECISION : 0;
            }
        }
    }

    /**
     * @dev Menghitung trace (jumlah elemen diagonal) matriks
     * @param matrix Matriks input
     * @return trace Nilai trace matriks
     */
    function trace(
        uint256[][] memory matrix
    ) internal pure returns (uint256 trace) {
        uint256 rows = matrix.length;
        require(rows > 0, "Matrix: matrix cannot be empty");
        uint256 cols = matrix[0].length;
        require(cols > 0, "Matrix: matrix must have columns");
        require(rows == cols, "Matrix: matrix must be square for trace");
        
        trace = 0;
        for (uint256 i = 0; i < rows; i++) {
            trace += matrix[i][i];
        }
    }

    /**
     * @dev Menghitung determinan matriks 2x2
     * @param matrix Matriks 2x2
     * @return determinant Nilai determinan
     */
    function determinant2x2(
        uint256[][] memory matrix
    ) internal pure returns (int256 determinant) {
        require(matrix.length == 2, "Matrix: matrix must be 2x2");
        require(matrix[0].length == 2, "Matrix: matrix must be 2x2");
        
        int256 a = int256(matrix[0][0]);
        int256 b = int256(matrix[0][1]);
        int256 c = int256(matrix[1][0]);
        int256 d = int256(matrix[1][1]);
        
        determinant = (a * d - b * c) / int256(PRECISION);
    }

    /**
     * @dev Menghitung determinan matriks 3x3
     * @param matrix Matriks 3x3
     * @return determinant Nilai determinan
     */
    function determinant3x3(
        uint256[][] memory matrix
    ) internal pure returns (int256 determinant) {
        require(matrix.length == 3, "Matrix: matrix must be 3x3");
        require(matrix[0].length == 3, "Matrix: matrix must be 3x3");
        
        int256 a = int256(matrix[0][0]);
        int256 b = int256(matrix[0][1]);
        int256 c = int256(matrix[0][2]);
        int256 d = int256(matrix[1][0]);
        int256 e = int256(matrix[1][1]);
        int256 f = int256(matrix[1][2]);
        int256 g = int256(matrix[2][0]);
        int256 h = int256(matrix[2][1]);
        int256 i = int256(matrix[2][2]);
        
        determinant = (a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)) / int256(PRECISION * PRECISION);
    }

    /**
     * @dev Melakukan LU decomposition sederhana untuk matriks 2x2 atau 3x3
     * @param matrix Matriks input
     * @return l Matriks lower triangular
     * @return u Matriks upper triangular
     */
    function luDecomposition(
        uint256[][] memory matrix
    ) internal pure returns (uint256[][] memory l, uint256[][] memory u) {
        uint256 n = matrix.length;
        require(n > 0 && n <= 3, "Matrix: LU decomposition only supported for 2x2 and 3x3 matrices");
        require(matrix[0].length == n, "Matrix: matrix must be square");
        
        l = identity(n);
        u = new uint256[][](n);
        
        for (uint256 i = 0; i < n; i++) {
            u[i] = new uint256[](n);
        }
        
        if (n == 2) {
            // LU decomposition untuk matriks 2x2
            u[0][0] = matrix[0][0];
            u[0][1] = matrix[0][1];
            
            if (u[0][0] != 0) {
                l[1][0] = (matrix[1][0] * PRECISION) / u[0][0];
            }
            
            u[1][1] = matrix[1][1] - (l[1][0] * u[0][1]) / PRECISION;
        } else if (n == 3) {
            // LU decomposition untuk matriks 3x3
            u[0][0] = matrix[0][0];
            u[0][1] = matrix[0][1];
            u[0][2] = matrix[0][2];
            
            if (u[0][0] != 0) {
                l[1][0] = (matrix[1][0] * PRECISION) / u[0][0];
                l[2][0] = (matrix[2][0] * PRECISION) / u[0][0];
            }
            
            u[1][1] = matrix[1][1] - (l[1][0] * u[0][1]) / PRECISION;
            u[1][2] = matrix[1][2] - (l[1][0] * u[0][2]) / PRECISION;
            
            if (u[1][1] != 0) {
                l[2][1] = ((matrix[2][1] * PRECISION) - (l[2][0] * u[0][1])) / u[1][1];
            }
            
            u[2][2] = matrix[2][2] - (l[2][0] * u[0][2] + l[2][1] * u[1][2]) / PRECISION;
        }
    }

    /**
     * @dev Menghitung rank matriks (perkiraan sederhana)
     * @param matrix Matriks input
     * @return rank Perkiraan rank matriks
     */
    function rank(
        uint256[][] memory matrix
    ) internal pure returns (uint256 rank) {
        uint256 rows = matrix.length;
        require(rows > 0, "Matrix: matrix cannot be empty");
        uint256 cols = matrix[0].length;
        require(cols > 0, "Matrix: matrix must have columns");
        
        // Implementasi sederhana: hitung jumlah baris yang tidak semua nol
        rank = 0;
        for (uint256 i = 0; i < rows; i++) {
            bool hasNonZero = false;
            for (uint256 j = 0; j < cols; j++) {
                if (matrix[i][j] > 0) {
                    hasNonZero = true;
                    break;
                }
            }
            if (hasNonZero) {
                rank++;
            }
        }
    }

    /**
     * @dev Mengekstrak submatrix dari matriks
     * @param matrix Matriks input
     * @param startRow Baris awal
     * @param endRow Baris akhir
     * @param startCol Kolom awal
     * @param endCol Kolom akhir
     * @return submatrix Submatrix yang diekstrak
     */
    function submatrix(
        uint256[][] memory matrix,
        uint256 startRow,
        uint256 endRow,
        uint256 startCol,
        uint256 endCol
    ) internal pure returns (uint256[][] memory submatrix) {
        uint256 rows = matrix.length;
        require(rows > 0, "Matrix: matrix cannot be empty");
        uint256 cols = matrix[0].length;
        require(cols > 0, "Matrix: matrix must have columns");
        
        require(startRow < endRow, "Matrix: invalid row range");
        require(startCol < endCol, "Matrix: invalid column range");
        require(endRow <= rows, "Matrix: endRow out of bounds");
        require(endCol <= cols, "Matrix: endCol out of bounds");
        
        uint256 subRows = endRow - startRow;
        uint256 subCols = endCol - startCol;
        
        submatrix = new uint256[][](subRows);
        for (uint256 i = 0; i < subRows; i++) {
            submatrix[i] = new uint256[](subCols);
            for (uint256 j = 0; j < subCols; j++) {
                submatrix[i][j] = matrix[startRow + i][startCol + j];
            }
        }
    }

    /**
     * @dev Memvalidasi matriks untuk operasi
     * @param matrix Matriks yang akan divalidasi
     * @return isValid True jika matriks valid
     * @return errorMessage Pesan error jika tidak valid
     */
    function validateMatrix(
        uint256[][] memory matrix
    ) internal pure returns (bool isValid, string memory errorMessage) {
        if (matrix.length == 0) {
            return (false, "Matrix: matrix cannot be empty");
        }
        
        uint256 cols = matrix[0].length;
        if (cols == 0) {
            return (false, "Matrix: matrix must have columns");
        }
        
        // Cek konsistensi jumlah kolom
        for (uint256 i = 1; i < matrix.length; i++) {
            if (matrix[i].length != cols) {
                return (false, "Matrix: inconsistent number of columns");
            }
        }
        
        // Cek untuk overflow potential
        for (uint256 i = 0; i < matrix.length; i++) {
            for (uint256 j = 0; j < cols; j++) {
                if (matrix[i][j] > type(uint256).max / PRECISION) {
                    return (false, "Matrix: value too large for fixed-point operations");
                }
            }
        }
        
        return (true, "");
    }

    /**
     * @dev Mengonversi matriks ke array 1D (row-major order)
     * @param matrix Matriks input
     * @return flattened Array 1D hasil flattening
     */
    function flatten(
        uint256[][] memory matrix
    ) internal pure returns (uint256[] memory flattened) {
        uint256 rows = matrix.length;
        require(rows > 0, "Matrix: matrix cannot be empty");
        uint256 cols = matrix[0].length;
        require(cols > 0, "Matrix: matrix must have columns");
        
        flattened = new uint256[](rows * cols);
        uint256 index = 0;
        
        for (uint256 i = 0; i < rows; i++) {
            for (uint256 j = 0; j < cols; j++) {
                flattened[index++] = matrix[i][j];
            }
        }
    }

    /**
     * @dev Mengonversi array 1D ke matriks dengan dimensi tertentu
     * @param array Array input 1D
     * @param rows Jumlah baris
     * @param cols Jumlah kolom
     * @return matrix Matriks hasil konversi
     */
    function unflatten(
        uint256[] memory array,
        uint256 rows,
        uint256 cols
    ) internal pure returns (uint256[][] memory matrix) {
        require(array.length == rows * cols, "Matrix: array length must match matrix dimensions");
        
        matrix = new uint256[][](rows);
        uint256 index = 0;
        
        for (uint256 i = 0; i < rows; i++) {
            matrix[i] = new uint256[](cols);
            for (uint256 j = 0; j < cols; j++) {
                matrix[i][j] = array[index++];
            }
        }
    }
}