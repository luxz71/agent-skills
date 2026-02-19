// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ModelSerializer
 * @dev Library untuk serialisasi dan deserialisasi model machine learning dengan fixed-point arithmetic
 * @notice Library ini menyediakan fungsi untuk menyimpan dan memuat parameter model dengan optimasi storage
 * @author Rizky Reza
 */
library ModelSerializer {
    // Precision untuk fixed-point arithmetic (18 decimal places)
    uint256 private constant PRECISION = 1e18;
    
    // Versi format serialisasi
    uint256 private constant SERIALIZATION_VERSION = 1;

    /**
     * @dev Struktur untuk metadata model
     */
    struct ModelMetadata {
        string modelName;
        string version;
        uint256 inputSize;
        uint256 outputSize;
        uint256 parameterCount;
        uint256 serializationVersion;
        uint256 timestamp;
        string modelType;
    }

    /**
     * @dev Struktur untuk serialized model
     */
    struct SerializedModel {
        ModelMetadata metadata;
        uint256[] parameters;
        bytes32 checksum;
    }

    /**
     * @dev Serialisasi parameter model menjadi format yang optimal untuk storage
     * @param parameters Array parameter model
     * @param metadata Metadata model
     * @param timestamp Timestamp untuk model (opsional, bisa 0)
     * @return serializedData Data model yang sudah diserialisasi
     */
    function serializeModel(
        uint256[] memory parameters,
        ModelMetadata memory metadata,
        uint256 timestamp
    ) internal pure returns (SerializedModel memory serializedData) {
        require(parameters.length > 0, "ModelSerializer: parameters cannot be empty");
        require(bytes(metadata.modelName).length > 0, "ModelSerializer: model name cannot be empty");
        
        // Set metadata
        metadata.parameterCount = parameters.length;
        metadata.serializationVersion = SERIALIZATION_VERSION;
        metadata.timestamp = timestamp;
        
        // Copy parameters
        serializedData.parameters = new uint256[](parameters.length);
        for (uint256 i = 0; i < parameters.length; i++) {
            serializedData.parameters[i] = parameters[i];
        }
        
        // Set metadata
        serializedData.metadata = metadata;
        
        // Calculate checksum
        serializedData.checksum = calculateChecksum(parameters, metadata);
    }

    /**
     * @dev Deserialisasi data model menjadi parameter asli
     * @param serializedData Data model yang sudah diserialisasi
     * @return parameters Array parameter model
     * @return metadata Metadata model
     */
    function deserializeModel(
        SerializedModel memory serializedData
    ) internal pure returns (
        uint256[] memory parameters,
        ModelMetadata memory metadata
    ) {
        // Validasi checksum
        require(
            validateChecksum(serializedData),
            "ModelSerializer: checksum validation failed"
        );
        
        // Validasi versi
        require(
            serializedData.metadata.serializationVersion == SERIALIZATION_VERSION,
            "ModelSerializer: incompatible serialization version"
        );
        
        // Copy parameters
        parameters = new uint256[](serializedData.parameters.length);
        for (uint256 i = 0; i < serializedData.parameters.length; i++) {
            parameters[i] = serializedData.parameters[i];
        }
        
        // Copy metadata
        metadata = serializedData.metadata;
    }

    /**
     * @dev Menghitung checksum untuk validasi integritas data
     * @param parameters Array parameter model
     * @param metadata Metadata model
     * @return checksum Hash checksum
     */
    function calculateChecksum(
        uint256[] memory parameters,
        ModelMetadata memory metadata
    ) internal pure returns (bytes32 checksum) {
        bytes memory data = abi.encodePacked(
            metadata.modelName,
            metadata.version,
            metadata.inputSize,
            metadata.outputSize,
            metadata.parameterCount,
            metadata.serializationVersion,
            metadata.timestamp,
            metadata.modelType
        );
        
        for (uint256 i = 0; i < parameters.length; i++) {
            data = abi.encodePacked(data, parameters[i]);
        }
        
        checksum = keccak256(data);
    }

    /**
     * @dev Memvalidasi checksum data model
     * @param serializedData Data model yang akan divalidasi
     * @return isValid True jika checksum valid
     */
    function validateChecksum(
        SerializedModel memory serializedData
    ) internal pure returns (bool isValid) {
        bytes32 computedChecksum = calculateChecksum(
            serializedData.parameters,
            serializedData.metadata
        );
        
        isValid = (computedChecksum == serializedData.checksum);
    }

    /**
     * @dev Kompresi parameter model menggunakan teknik sederhana
     * @param parameters Array parameter model
     * @param compressionType Tipe kompresi (0: none, 1: delta encoding)
     * @return compressedParameters Parameter yang sudah dikompres
     * @return compressionInfo Informasi kompresi
     */
    function compressParameters(
        uint256[] memory parameters,
        uint256 compressionType
    ) internal pure returns (
        uint256[] memory compressedParameters,
        uint256[] memory compressionInfo
    ) {
        require(parameters.length > 0, "ModelSerializer: parameters cannot be empty");
        
        if (compressionType == 0) {
            // No compression
            compressedParameters = parameters;
            compressionInfo = new uint256[](1);
            compressionInfo[0] = 0; // Compression type
        } else if (compressionType == 1) {
            // Delta encoding
            compressedParameters = new uint256[](parameters.length);
            compressionInfo = new uint256[](2);
            
            compressedParameters[0] = parameters[0]; // Store first value as-is
            uint256 minDelta = type(uint256).max;
            uint256 maxDelta = 0;
            
            for (uint256 i = 1; i < parameters.length; i++) {
                uint256 delta;
                if (parameters[i] >= parameters[i - 1]) {
                    delta = parameters[i] - parameters[i - 1];
                } else {
                    delta = parameters[i - 1] - parameters[i];
                    // Use high bit to indicate negative delta
                    delta = delta | (1 << 255);
                }
                
                compressedParameters[i] = delta;
                
                if (delta < minDelta) {
                    minDelta = delta;
                }
                if (delta > maxDelta) {
                    maxDelta = delta;
                }
            }
            
            compressionInfo[0] = 1; // Compression type
            compressionInfo[1] = minDelta; // Additional info for decompression
        } else {
            revert("ModelSerializer: unsupported compression type");
        }
    }

    /**
     * @dev Dekompresi parameter model
     * @param compressedParameters Parameter yang sudah dikompres
     * @param compressionInfo Informasi kompresi
     * @return parameters Parameter yang sudah didekompres
     */
    function decompressParameters(
        uint256[] memory compressedParameters,
        uint256[] memory compressionInfo
    ) internal pure returns (uint256[] memory parameters) {
        require(compressedParameters.length > 0, "ModelSerializer: compressed parameters cannot be empty");
        require(compressionInfo.length > 0, "ModelSerializer: compression info cannot be empty");
        
        uint256 compressionType = compressionInfo[0];
        
        if (compressionType == 0) {
            // No compression
            parameters = compressedParameters;
        } else if (compressionType == 1) {
            // Delta encoding
            parameters = new uint256[](compressedParameters.length);
            parameters[0] = compressedParameters[0];
            
            for (uint256 i = 1; i < compressedParameters.length; i++) {
                uint256 delta = compressedParameters[i];
                
                if ((delta >> 255) & 1 == 1) {
                    // Negative delta
                    parameters[i] = parameters[i - 1] - (delta & ((1 << 255) - 1));
                } else {
                    // Positive delta
                    parameters[i] = parameters[i - 1] + delta;
                }
            }
        } else {
            revert("ModelSerializer: unsupported compression type");
        }
    }

    /**
     * @dev Quantisasi parameter untuk mengurangi storage requirements
     * @param parameters Array parameter model
     * @param quantizationLevels Jumlah level quantisasi
     * @return quantizedParameters Parameter yang sudah diquantisasi
     * @return quantizationTable Tabel quantisasi
     */
    function quantizeParameters(
        uint256[] memory parameters,
        uint256 quantizationLevels
    ) internal pure returns (
        uint256[] memory quantizedParameters,
        uint256[] memory quantizationTable
    ) {
        require(parameters.length > 0, "ModelSerializer: parameters cannot be empty");
        require(quantizationLevels > 1, "ModelSerializer: quantization levels must be greater than 1");
        
        // Find min and max values
        uint256 minVal = parameters[0];
        uint256 maxVal = parameters[0];
        
        for (uint256 i = 1; i < parameters.length; i++) {
            if (parameters[i] < minVal) {
                minVal = parameters[i];
            }
            if (parameters[i] > maxVal) {
                maxVal = parameters[i];
            }
        }
        
        // Create quantization table
        quantizationTable = new uint256[](quantizationLevels);
        uint256 range = maxVal - minVal;
        
        for (uint256 i = 0; i < quantizationLevels; i++) {
            quantizationTable[i] = minVal + (range * i) / (quantizationLevels - 1);
        }
        
        // Quantize parameters
        quantizedParameters = new uint256[](parameters.length);
        
        for (uint256 i = 0; i < parameters.length; i++) {
            // Find closest quantization level
            uint256 bestLevel = 0;
            uint256 bestDiff = type(uint256).max;
            
            for (uint256 j = 0; j < quantizationLevels; j++) {
                uint256 diff = parameters[i] > quantizationTable[j] ? 
                    parameters[i] - quantizationTable[j] : 
                    quantizationTable[j] - parameters[i];
                
                if (diff < bestDiff) {
                    bestDiff = diff;
                    bestLevel = j;
                }
            }
            
            quantizedParameters[i] = bestLevel;
        }
    }

    /**
     * @dev Dequantisasi parameter
     * @param quantizedParameters Parameter yang sudah diquantisasi
     * @param quantizationTable Tabel quantisasi
     * @return parameters Parameter yang sudah didequantisasi
     */
    function dequantizeParameters(
        uint256[] memory quantizedParameters,
        uint256[] memory quantizationTable
    ) internal pure returns (uint256[] memory parameters) {
        require(quantizedParameters.length > 0, "ModelSerializer: quantized parameters cannot be empty");
        require(quantizationTable.length > 0, "ModelSerializer: quantization table cannot be empty");
        
        parameters = new uint256[](quantizedParameters.length);
        
        for (uint256 i = 0; i < quantizedParameters.length; i++) {
            require(
                quantizedParameters[i] < quantizationTable.length,
                "ModelSerializer: invalid quantization level"
            );
            
            parameters[i] = quantizationTable[quantizedParameters[i]];
        }
    }

    /**
     * @dev Serialisasi model ke format bytes untuk external storage
     * @param serializedModel Model yang sudah diserialisasi
     * @return data Data model dalam format bytes
     */
    function toBytes(
        SerializedModel memory serializedModel
    ) internal pure returns (bytes memory data) {
        data = abi.encode(
            serializedModel.metadata.modelName,
            serializedModel.metadata.version,
            serializedModel.metadata.inputSize,
            serializedModel.metadata.outputSize,
            serializedModel.metadata.parameterCount,
            serializedModel.metadata.serializationVersion,
            serializedModel.metadata.timestamp,
            serializedModel.metadata.modelType,
            serializedModel.parameters,
            serializedModel.checksum
        );
    }

    /**
     * @dev Deserialisasi model dari format bytes
     * @param data Data model dalam format bytes
     * @return serializedModel Model yang sudah dideserialisasi
     */
    function fromBytes(
        bytes memory data
    ) internal pure returns (SerializedModel memory serializedModel) {
        (
            string memory modelName,
            string memory version,
            uint256 inputSize,
            uint256 outputSize,
            uint256 parameterCount,
            uint256 serializationVersion,
            uint256 timestamp,
            string memory modelType,
            uint256[] memory parameters,
            bytes32 checksum
        ) = abi.decode(
            data,
            (string, string, uint256, uint256, uint256, uint256, uint256, string, uint256[], bytes32)
        );
        
        serializedModel.metadata = ModelMetadata({
            modelName: modelName,
            version: version,
            inputSize: inputSize,
            outputSize: outputSize,
            parameterCount: parameterCount,
            serializationVersion: serializationVersion,
            timestamp: timestamp,
            modelType: modelType
        });
        
        serializedModel.parameters = parameters;
        serializedModel.checksum = checksum;
    }

    /**
     * @dev Memvalidasi metadata model
     * @param metadata Metadata model
     * @return isValid True jika metadata valid
     * @return errorMessage Pesan error jika tidak valid
     */
    function validateMetadata(
        ModelMetadata memory metadata
    ) internal pure returns (bool isValid, string memory errorMessage) {
        if (bytes(metadata.modelName).length == 0) {
            return (false, "ModelSerializer: model name cannot be empty");
        }
        
        if (metadata.inputSize == 0) {
            return (false, "ModelSerializer: input size must be positive");
        }
        
        if (metadata.outputSize == 0) {
            return (false, "ModelSerializer: output size must be positive");
        }
        
        if (metadata.parameterCount == 0) {
            return (false, "ModelSerializer: parameter count must be positive");
        }
        
        if (metadata.serializationVersion != SERIALIZATION_VERSION) {
            return (false, "ModelSerializer: incompatible serialization version");
        }
        
        return (true, "");
    }

    /**
     * @dev Memvalidasi parameter model
     * @param parameters Parameter model
     * @return isValid True jika parameter valid
     * @return errorMessage Pesan error jika tidak valid
     */
    function validateParameters(
        uint256[] memory parameters
    ) internal pure returns (bool isValid, string memory errorMessage) {
        if (parameters.length == 0) {
            return (false, "ModelSerializer: parameters cannot be empty");
        }
        
        // Cek untuk nilai yang terlalu besar
        for (uint256 i = 0; i < parameters.length; i++) {
            if (parameters[i] > type(uint256).max / PRECISION) {
                return (false, "ModelSerializer: parameter value too large for fixed-point operations");
            }
        }
        
        return (true, "");
    }

    /**
     * @dev Membandingkan dua model untuk kompatibilitas
     * @param model1 Model pertama
     * @param model2 Model kedua
     * @return isCompatible True jika model kompatibel
     * @return compatibilityInfo Informasi kompatibilitas
     */
    function checkCompatibility(
        SerializedModel memory model1,
        SerializedModel memory model2
    ) internal pure returns (
        bool isCompatible,
        string memory compatibilityInfo
    ) {
        // Check model type
        if (keccak256(bytes(model1.metadata.modelType)) != keccak256(bytes(model2.metadata.modelType))) {
            return (false, "Model types are different");
        }
        
        // Check input/output sizes
        if (model1.metadata.inputSize != model2.metadata.inputSize) {
            return (false, "Input sizes are different");
        }
        
        if (model1.metadata.outputSize != model2.metadata.outputSize) {
            return (false, "Output sizes are different");
        }
        
        // Check parameter counts
        if (model1.metadata.parameterCount != model2.metadata.parameterCount) {
            return (false, "Parameter counts are different");
        }
        
        return (true, "Models are compatible");
    }

    /**
     * @dev Mengembalikan ukuran storage yang diperlukan untuk model
     * @param serializedModel Model yang sudah diserialisasi
     * @return storageSize Ukuran storage dalam bytes
     */
    function getStorageSize(
        SerializedModel memory serializedModel
    ) internal pure returns (uint256 storageSize) {
        // Hitung ukuran metadata
        storageSize += bytes(serializedModel.metadata.modelName).length;
        storageSize += bytes(serializedModel.metadata.version).length;
        storageSize += bytes(serializedModel.metadata.modelType).length;
        storageSize += 7 * 32; // 7 uint256 fields
        
        // Hitung ukuran parameters
        storageSize += serializedModel.parameters.length * 32;
        
        // Tambahan untuk checksum
        storageSize += 32;
    }
}