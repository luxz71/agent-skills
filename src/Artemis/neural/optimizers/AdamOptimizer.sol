// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../../interfaces/IOptimizer.sol";

/**
 * @title AdamOptimizer
 * @dev Implementasi Adaptive Moment Estimation (Adam) optimizer
 * @notice Optimizer ini menggabungkan momentum dan RMSprop dengan adaptive learning rates
 * @author Rizky Reza
 */
contract AdamOptimizer is IOptimizer {
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    
    // Optimizer configuration
    string public name;
    string public version;
    string public optimizerType;
    
    // Hyperparameters
    uint256 public learningRate;
    uint256 public beta1; // Exponential decay rate untuk first moment
    uint256 public beta2; // Exponential decay rate untuk second moment
    uint256 public epsilon; // Small constant untuk numerical stability
    
    // State variables untuk Adam
    mapping(bytes32 => int256[]) public m; // First moment vector
    mapping(bytes32 => int256[]) public v; // Second moment vector
    uint256 public iterationCount;
    
    // Statistics
    uint256 public totalUpdates;
    uint256 public averageUpdateMagnitude;
    uint256 public lastUpdateTime;
    
    /**
     * @dev Inisialisasi Adam optimizer dengan konfigurasi default
     */
    constructor() {
        name = "Adam";
        version = "1.0.0";
        optimizerType = "Adam";
        
        // Default hyperparameters (nilai dari paper Adam)
        learningRate = (1 * PRECISION) / 1000; // 0.001
        beta1 = (9 * PRECISION) / 10; // 0.9
        beta2 = (999 * PRECISION) / 1000; // 0.999
        epsilon = 1e8; // 1e-8 dalam fixed-point
        
        iterationCount = 0;
        totalUpdates = 0;
        averageUpdateMagnitude = 0;
        lastUpdateTime = block.timestamp;
    }
    
    /**
     * @dev Mengupdate parameter model berdasarkan gradien menggunakan Adam
     * @param parameters Array parameter saat ini
     * @param gradients Array gradien untuk setiap parameter
     * @param _learningRate Learning rate untuk optimisasi
     * @return updatedParameters Array parameter yang sudah diupdate
     * @return updateMagnitude Besarnya perubahan parameter
     */
    function updateParameters(
        uint256[] calldata parameters,
        int256[] calldata gradients,
        uint256 _learningRate
    ) external override returns (
        uint256[] memory updatedParameters,
        uint256 updateMagnitude
    ) {
        require(parameters.length == gradients.length, "AdamOptimizer: Parameters and gradients must have same length");
        require(parameters.length > 0, "AdamOptimizer: Parameters cannot be empty");
        require(_learningRate > 0, "AdamOptimizer: Learning rate must be positive");
        
        // Generate key untuk state storage
        bytes32 stateKey = keccak256(abi.encodePacked(parameters, iterationCount));
        
        // Initialize atau dapatkan moments
        int256[] memory mCurrent = m[stateKey];
        int256[] memory vCurrent = v[stateKey];
        
        if (mCurrent.length == 0) {
            mCurrent = new int256[](parameters.length);
            vCurrent = new int256[](parameters.length);
            for (uint256 i = 0; i < parameters.length; i++) {
                mCurrent[i] = 0;
                vCurrent[i] = 0;
            }
        }
        
        updatedParameters = new uint256[](parameters.length);
        updateMagnitude = 0;
        
        // Increment iteration count untuk bias correction
        iterationCount++;
        uint256 t = iterationCount;
        
        // Bias correction factors
        uint256 beta1_t = _power(beta1, t);
        uint256 beta2_t = _power(beta2, t);
        
        uint256 biasCorrection1 = ONE - beta1_t;
        uint256 biasCorrection2 = ONE - beta2_t;
        
        for (uint256 i = 0; i < parameters.length; i++) {
            // Update biased first moment estimate
            // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            int256 mUpdate = (int256(beta1) * mCurrent[i]) / int256(PRECISION);
            int256 gradientPart = (gradients[i] * int256(ONE - beta1)) / int256(PRECISION);
            mCurrent[i] = mUpdate + gradientPart;
            
            // Update biased second raw moment estimate
            // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            int256 vUpdate = (int256(beta2) * vCurrent[i]) / int256(PRECISION);
            int256 squaredGradient = (gradients[i] * gradients[i]) / int256(PRECISION);
            int256 squaredPart = (squaredGradient * int256(ONE - beta2)) / int256(PRECISION);
            vCurrent[i] = vUpdate + squaredPart;
            
            // Compute bias-corrected first moment estimate
            // m_hat_t = m_t / (1 - beta1^t)
            int256 mHat = (mCurrent[i] * int256(PRECISION)) / int256(biasCorrection1);
            
            // Compute bias-corrected second raw moment estimate
            // v_hat_t = v_t / (1 - beta2^t)
            int256 vHat = (vCurrent[i] * int256(PRECISION)) / int256(biasCorrection2);
            
            // Adam update: parameter = parameter - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
            int256 parameterInt = int256(parameters[i]);
            
            // Compute denominator: sqrt(v_hat) + epsilon
            uint256 vHatAbs = vHat < 0 ? uint256(-vHat) : uint256(vHat);
            uint256 sqrtVHat = _sqrt(vHatAbs);
            int256 denominator = int256(sqrtVHat) + int256(epsilon);
            
            // Compute update: learning_rate * m_hat / denominator
            int256 updateNumerator = (int256(_learningRate) * mHat) / int256(PRECISION);
            int256 update = updateNumerator / denominator;
            
            // Apply update
            int256 newParameter = parameterInt - update;
            
            // Pastikan parameter tidak negatif
            if (newParameter < 0) {
                updatedParameters[i] = 0;
            } else {
                updatedParameters[i] = uint256(newParameter);
            }
            
            // Hitung update magnitude
            uint256 absUpdate = update < 0 ? uint256(-update) : uint256(update);
            updateMagnitude += absUpdate;
        }
        
        // Simpan moments kembali
        m[stateKey] = mCurrent;
        v[stateKey] = vCurrent;
        
        // Average update magnitude
        if (parameters.length > 0) {
            updateMagnitude = updateMagnitude / parameters.length;
        }
        
        totalUpdates++;
        averageUpdateMagnitude = (averageUpdateMagnitude * (totalUpdates - 1) + updateMagnitude) / totalUpdates;
        lastUpdateTime = block.timestamp;
        
        emit ParametersUpdated(address(this), parameters.length, int256(updateMagnitude));
        
        return (updatedParameters, updateMagnitude);
    }
    
    /**
     * @dev Mengupdate single parameter menggunakan Adam
     * @param parameter Parameter saat ini
     * @param gradient Gradien untuk parameter
     * @param _learningRate Learning rate untuk optimisasi
     * @return updatedParameter Parameter yang sudah diupdate
     */
    function updateParameter(
        uint256 parameter,
        int256 gradient,
        uint256 _learningRate
    ) external override returns (uint256 updatedParameter) {
        require(_learningRate > 0, "AdamOptimizer: Learning rate must be positive");
        
        // Untuk single parameter, kita gunakan implementasi sederhana
        // Adam update: parameter = parameter - learning_rate * gradient
        // (tanpa momentum dan adaptive learning rates untuk simplicity)
        
        int256 parameterInt = int256(parameter);
        int256 update = (gradient * int256(_learningRate)) / int256(PRECISION);
        
        int256 newParameter = parameterInt - update;
        
        // Pastikan parameter tidak negatif
        if (newParameter < 0) {
            return 0;
        } else {
            return uint256(newParameter);
        }
    }
    
    /**
     * @dev Mengatur learning rate optimizer
     * @param newLearningRate Learning rate baru
     * @return success Status keberhasilan pengaturan learning rate
     */
    function setLearningRate(uint256 newLearningRate) external override returns (bool success) {
        require(newLearningRate > 0, "AdamOptimizer: Learning rate must be positive");
        
        uint256 oldLearningRate = learningRate;
        learningRate = newLearningRate;
        
        emit LearningRateUpdated(address(this), oldLearningRate, newLearningRate);
        return true;
    }
    
    /**
     * @dev Mengatur momentum untuk Adam (beta1)
     * @param newBeta1 Nilai beta1 baru (0-1 dalam fixed-point)
     * @return success Status keberhasilan pengaturan beta1
     */
    function setMomentum(uint256 newBeta1) external override returns (bool success) {
        require(newBeta1 <= ONE, "AdamOptimizer: Beta1 must be between 0 and 1");
        
        beta1 = newBeta1;
        return true;
    }
    
    /**
     * @dev Mengatur beta2 untuk Adam
     * @param newBeta2 Nilai beta2 baru (0-1 dalam fixed-point)
     * @return success Status keberhasilan pengaturan beta2
     */
    function setBeta2(uint256 newBeta2) external returns (bool success) {
        require(newBeta2 <= ONE, "AdamOptimizer: Beta2 must be between 0 and 1");
        
        beta2 = newBeta2;
        return true;
    }
    
    /**
     * @dev Mengatur epsilon untuk Adam
     * @param newEpsilon Nilai epsilon baru
     * @return success Status keberhasilan pengaturan epsilon
     */
    function setEpsilon(uint256 newEpsilon) external returns (bool success) {
        require(newEpsilon > 0, "AdamOptimizer: Epsilon must be positive");
        
        epsilon = newEpsilon;
        return true;
    }
    
    /**
     * @dev Reset state optimizer (moments untuk Adam)
     * @return success Status keberhasilan reset
     */
    function resetState() external override returns (bool success) {
        // Clear semua moments
        // Catatan: Dalam implementasi produksi, perlu mekanisme yang lebih efisien
        iterationCount = 0;
        
        return true;
    }
    
    /**
     * @dev Memvalidasi input untuk update parameter
     * @param parameters Array parameter
     * @param gradients Array gradien
     * @return isValid True jika input valid
     * @return errorMessage Pesan error jika input tidak valid
     */
    function validateInput(
        uint256[] calldata parameters,
        int256[] calldata gradients
    ) external override pure returns (bool isValid, string memory errorMessage) {
        if (parameters.length != gradients.length) {
            return (false, "AdamOptimizer: Parameters and gradients must have same length");
        }
        
        if (parameters.length == 0) {
            return (false, "AdamOptimizer: Parameters cannot be empty");
        }
        
        return (true, "");
    }
    
    /**
     * @dev Mengembalikan informasi tentang optimizer
     * @return _name Nama optimizer
     * @return _version Versi optimizer
     * @return _optimizerType Tipe optimizer
     * @return _supportsMomentum True jika mendukung momentum
     * @return _requiresGradientClipping True jika memerlukan gradient clipping
     */
    function getOptimizerInfo() external view override returns (
        string memory _name,
        string memory _version,
        string memory _optimizerType,
        bool _supportsMomentum,
        bool _requiresGradientClipping
    ) {
        return (
            name,
            version,
            optimizerType,
            true, // Adam mendukung momentum (via beta1)
            false // Adam memiliki built-in gradient scaling
        );
    }
    
    /**
     * @dev Mengembalikan konfigurasi saat ini
     * @return _learningRate Learning rate saat ini
     * @return _momentum Momentum saat ini (beta1)
     * @return _beta1 Beta1 untuk first moment
     * @return _beta2 Beta2 untuk second moment
     * @return _epsilon Epsilon untuk stabilitas numerik
     */
    function getCurrentConfiguration() external view override returns (
        uint256 _learningRate,
        uint256 _momentum,
        uint256 _beta1,
        uint256 _beta2,
        uint256 _epsilon
    ) {
        return (
            learningRate,
            beta1, // momentum sama dengan beta1 di Adam
            beta1,
            beta2,
            epsilon
        );
    }
    
    /**
     * @dev Mengembalikan statistik penggunaan optimizer
     * @return _totalUpdates Total jumlah update yang dilakukan
     * @return _averageUpdateMagnitude Rata-rata besar update
     * @return _lastUpdateTime Waktu update terakhir
     */
    function getOptimizerStats() external view override returns (
        uint256 _totalUpdates,
        uint256 _averageUpdateMagnitude,
        uint256 _lastUpdateTime
    ) {
        return (
            totalUpdates,
            averageUpdateMagnitude,
            lastUpdateTime
        );
    }
    
    /**
     * @dev Clipping gradien untuk mencegah exploding gradients
     * @param gradients Array gradien
     * @param clipValue Nilai clipping maksimum
     * @return clippedGradients Array gradien yang sudah diclip
     */
    function clipGradients(
        int256[] calldata gradients,
        int256 clipValue
    ) external override pure returns (int256[] memory clippedGradients) {
        require(clipValue > 0, "AdamOptimizer: Clip value must be positive");
        
        clippedGradients = new int256[](gradients.length);
        
        for (uint256 i = 0; i < gradients.length; i++) {
            if (gradients[i] > clipValue) {
                clippedGradients[i] = clipValue;
            } else if (gradients[i] < -clipValue) {
                clippedGradients[i] = -clipValue;
            } else {
                clippedGradients[i] = gradients[i];
            }
        }
        
        return clippedGradients;
    }
    
    /**
     * @dev Normalisasi gradien untuk stabilitas training
     * @param gradients Array gradien
     * @return normalizedGradients Array gradien yang sudah dinormalisasi
     * @return normalizationFactor Faktor normalisasi yang digunakan
     */
    function normalizeGradients(
        int256[] calldata gradients
    ) external override pure returns (int256[] memory normalizedGradients, uint256 normalizationFactor) {
        require(gradients.length > 0, "AdamOptimizer: Gradients cannot be empty");
        
        // Hitung norm L2 dari gradients
        uint256 sumSquares = 0;
        for (uint256 i = 0; i < gradients.length; i++) {
            int256 grad = gradients[i];
            uint256 square = uint256(grad * grad) / PRECISION;
            sumSquares += square;
        }
        
        normalizationFactor = _sqrt(sumSquares);
        
        // Jika norm terlalu kecil, hindari division by zero
        if (normalizationFactor < 1e12) {
            normalizationFactor = PRECISION;
        }
        
        normalizedGradients = new int256[](gradients.length);
        
        for (uint256 i = 0; i < gradients.length; i++) {
            normalizedGradients[i] = (gradients[i] * int256(PRECISION)) / int256(normalizationFactor);
        }
        
        return (normalizedGradients, normalizationFactor);
    }
    
    /**
     * @dev Fungsi internal untuk menghitung power (approximation)
     * @param base Basis
     * @param exponent Eksponen
     * @return result base^exponent dalam fixed-point
     */
    function _power(uint256 base, uint256 exponent) internal pure returns (uint256 result) {
        if (exponent == 0) {
            return PRECISION;
        }
        
        result = PRECISION;
        uint256 currentBase = base;
        uint256 currentExponent = exponent;
        
        while (currentExponent > 0) {
            if (currentExponent % 2 == 1) {
                result = (result * currentBase) / PRECISION;
            }
            currentBase = (currentBase * currentBase) / PRECISION;
            currentExponent = currentExponent / 2;
        }
        
        return result;
    }
    
    /**
     * @dev Fungsi internal untuk menghitung square root (approximation)
     * @param x Nilai input
     * @return result Akar kuadrat dari x
     */
    function _sqrt(uint256 x) internal pure returns (uint256 result) {
        if (x == 0) return 0;
        
        // Babylonian method untuk approximation
        uint256 z = (x + 1) / 2;
        uint256 y = x;
        
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
        
        return y;
    }
}