// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "../../interfaces/IOptimizer.sol";

/**
 * @title SGDOptimizer
 * @dev Implementasi Stochastic Gradient Descent optimizer dengan momentum support
 * @notice Optimizer ini menggunakan gradient descent dengan momentum untuk mempercepat konvergensi
 * @author Rizky Reza
 */
contract SGDOptimizer is IOptimizer {
    uint256 private constant PRECISION = 1e18;
    uint256 private constant ONE = 1e18;
    
    // Optimizer configuration
    string public name;
    string public version;
    string public optimizerType;
    
    // Hyperparameters
    uint256 public learningRate;
    uint256 public momentum;
    uint256 public learningRateDecay;
    
    // State variables untuk momentum
    mapping(bytes32 => int256[]) public velocities; // Velocity untuk setiap parameter set
    uint256 public iterationCount;
    
    // Statistics
    uint256 public totalUpdates;
    uint256 public averageUpdateMagnitude;
    uint256 public lastUpdateTime;
    
    /**
     * @dev Inisialisasi SGD optimizer dengan konfigurasi default
     */
    constructor() {
        name = "SGD";
        version = "1.0.0";
        optimizerType = "SGD";
        
        // Default hyperparameters
        learningRate = (1 * PRECISION) / 100; // 0.01
        momentum = (9 * PRECISION) / 10; // 0.9
        learningRateDecay = 0;
        
        iterationCount = 0;
        totalUpdates = 0;
        averageUpdateMagnitude = 0;
        lastUpdateTime = block.timestamp;
    }
    
    /**
     * @dev Mengupdate parameter model berdasarkan gradien menggunakan SGD
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
    ) external override pure returns (
        uint256[] memory updatedParameters,
        uint256 updateMagnitude
    ) {
        require(parameters.length == gradients.length, "SGDOptimizer: Parameters and gradients must have same length");
        require(parameters.length > 0, "SGDOptimizer: Parameters cannot be empty");
        require(_learningRate > 0, "SGDOptimizer: Learning rate must be positive");
        
        updatedParameters = new uint256[](parameters.length);
        updateMagnitude = 0;
        
        for (uint256 i = 0; i < parameters.length; i++) {
            // SGD update: parameter = parameter - learning_rate * gradient
            int256 parameterInt = int256(parameters[i]);
            int256 update = (gradients[i] * int256(_learningRate)) / int256(PRECISION);
            
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
        
        // Average update magnitude
        if (parameters.length > 0) {
            updateMagnitude = updateMagnitude / parameters.length;
        }
        
        return (updatedParameters, updateMagnitude);
    }
    
    /**
     * @dev Mengupdate parameter dengan momentum
     * @param parameters Array parameter saat ini
     * @param gradients Array gradien untuk setiap parameter
     * @param _learningRate Learning rate untuk optimisasi
     * @return updatedParameters Array parameter yang sudah diupdate
     * @return updateMagnitude Besarnya perubahan parameter
     */
    function updateParametersWithMomentum(
        uint256[] calldata parameters,
        int256[] calldata gradients,
        uint256 _learningRate
    ) external returns (
        uint256[] memory updatedParameters,
        uint256 updateMagnitude
    ) {
        require(parameters.length == gradients.length, "SGDOptimizer: Parameters and gradients must have same length");
        require(parameters.length > 0, "SGDOptimizer: Parameters cannot be empty");
        require(_learningRate > 0, "SGDOptimizer: Learning rate must be positive");
        
        // Generate key untuk velocity storage
        bytes32 velocityKey = keccak256(abi.encodePacked(parameters, iterationCount));
        
        // Initialize atau dapatkan velocities
        int256[] memory velocity = velocities[velocityKey];
        if (velocity.length == 0) {
            velocity = new int256[](parameters.length);
            for (uint256 i = 0; i < parameters.length; i++) {
                velocity[i] = 0;
            }
        }
        
        updatedParameters = new uint256[](parameters.length);
        updateMagnitude = 0;
        
        for (uint256 i = 0; i < parameters.length; i++) {
            // Momentum update: velocity = momentum * velocity - learning_rate * gradient
            // parameter = parameter + velocity
            
            int256 velocityUpdate = (int256(momentum) * velocity[i]) / int256(PRECISION);
            int256 gradientUpdate = (gradients[i] * int256(_learningRate)) / int256(PRECISION);
            
            velocity[i] = velocityUpdate - gradientUpdate;
            
            int256 parameterInt = int256(parameters[i]);
            int256 newParameter = parameterInt + velocity[i];
            
            // Pastikan parameter tidak negatif
            if (newParameter < 0) {
                updatedParameters[i] = 0;
            } else {
                updatedParameters[i] = uint256(newParameter);
            }
            
            // Hitung update magnitude
            uint256 absUpdate = velocity[i] < 0 ? uint256(-velocity[i]) : uint256(velocity[i]);
            updateMagnitude += absUpdate;
        }
        
        // Simpan velocities kembali
        velocities[velocityKey] = velocity;
        
        // Average update magnitude
        if (parameters.length > 0) {
            updateMagnitude = updateMagnitude / parameters.length;
        }
        
        iterationCount++;
        totalUpdates++;
        averageUpdateMagnitude = (averageUpdateMagnitude * (totalUpdates - 1) + updateMagnitude) / totalUpdates;
        lastUpdateTime = block.timestamp;
        
        emit ParametersUpdated(address(this), parameters.length, int256(updateMagnitude));
        
        return (updatedParameters, updateMagnitude);
    }
    
    /**
     * @dev Mengupdate single parameter
     * @param parameter Parameter saat ini
     * @param gradient Gradien untuk parameter
     * @param _learningRate Learning rate untuk optimisasi
     * @return updatedParameter Parameter yang sudah diupdate
     */
    function updateParameter(
        uint256 parameter,
        int256 gradient,
        uint256 _learningRate
    ) external override pure returns (uint256 updatedParameter) {
        require(_learningRate > 0, "SGDOptimizer: Learning rate must be positive");
        
        // SGD update: parameter = parameter - learning_rate * gradient
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
        require(newLearningRate > 0, "SGDOptimizer: Learning rate must be positive");
        
        uint256 oldLearningRate = learningRate;
        learningRate = newLearningRate;
        
        emit LearningRateUpdated(address(this), oldLearningRate, newLearningRate);
        return true;
    }
    
    /**
     * @dev Mengatur momentum untuk SGD dengan momentum
     * @param newMomentum Nilai momentum baru (0-1 dalam fixed-point)
     * @return success Status keberhasilan pengaturan momentum
     */
    function setMomentum(uint256 newMomentum) external override returns (bool success) {
        require(newMomentum <= ONE, "SGDOptimizer: Momentum must be between 0 and 1");
        
        momentum = newMomentum;
        return true;
    }
    
    /**
     * @dev Mengatur learning rate decay
     * @param newDecay Nilai decay baru
     * @return success Status keberhasilan pengaturan decay
     */
    function setLearningRateDecay(uint256 newDecay) external returns (bool success) {
        learningRateDecay = newDecay;
        return true;
    }
    
    /**
     * @dev Reset state optimizer (velocities untuk momentum)
     * @return success Status keberhasilan reset
     */
    function resetState() external override returns (bool success) {
        // Clear semua velocities
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
            return (false, "SGDOptimizer: Parameters and gradients must have same length");
        }
        
        if (parameters.length == 0) {
            return (false, "SGDOptimizer: Parameters cannot be empty");
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
            true, // SGD mendukung momentum
            false // SGD tidak memerlukan gradient clipping khusus
        );
    }
    
    /**
     * @dev Mengembalikan konfigurasi saat ini
     * @return _learningRate Learning rate saat ini
     * @return _momentum Momentum saat ini
     * @return _beta1 Beta1 (tidak digunakan di SGD)
     * @return _beta2 Beta2 (tidak digunakan di SGD)
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
            momentum,
            0, // beta1 tidak digunakan di SGD
            0, // beta2 tidak digunakan di SGD
            1e8 // epsilon kecil untuk stabilitas
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
        require(clipValue > 0, "SGDOptimizer: Clip value must be positive");
        
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
        require(gradients.length > 0, "SGDOptimizer: Gradients cannot be empty");
        
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