// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DeepfakeAuth {
    struct MediaRecord {
        string fileHash;
        bool isDeepfake;
        uint256 confidenceScore;  // Stored as integer (multiply by 10000 to preserve 4 decimal places)
        address uploader;
        uint256 timestamp;
    }

    MediaRecord[] public records;
    mapping(uint256 => MediaRecord) private recordsMap; // For O(1) lookups
    mapping(string => bool) public hashExists;
    uint256 private nextId = 1;

    event MediaRegistered(
        string fileHash,
        bool isDeepfake,
        uint256 confidenceScore,
        address indexed uploader,
        uint256 timestamp
    );
    
    event RecordAdded(uint256 id, string fileHash); // Event for off-chain tracking

    function registerMedia(
        string memory fileHash,
        bool isDeepfake,
        uint256 confidenceScore
    ) public {
        require(bytes(fileHash).length > 0, "File hash cannot be empty");
        require(!hashExists[fileHash], "File hash already exists");

        uint256 currentId = nextId;
        
        MediaRecord memory newRecord = MediaRecord({
            fileHash: fileHash,
            isDeepfake: isDeepfake,
            confidenceScore: confidenceScore,
            uploader: msg.sender,
            timestamp: block.timestamp
        });

        // Store in both array (for iteration) and mapping (for direct access)
        records.push(newRecord);
        recordsMap[currentId] = newRecord;
        
        hashExists[fileHash] = true;

        unchecked {
            nextId++; // Prevent overflow checks for gas efficiency
        }

        emit MediaRegistered(fileHash, isDeepfake, confidenceScore, msg.sender, block.timestamp);
        emit RecordAdded(currentId, fileHash); // Emit event for tracking
    }

    function getRecord(uint256 id) public view returns (MediaRecord memory) {
        // O(1) lookup using mapping instead of linear search
        return recordsMap[id];
    }

    function getAllRecords() public view returns (MediaRecord[] memory) {
        return records;
    }

    // Paginated function to fetch records in chunks with input validation
    function getRecordsPaginated(
        uint256 startIndex,
        uint256 pageSize
    ) public view returns (MediaRecord[] memory) {
        require(pageSize > 0, "Page size must be greater than 0");
        require(pageSize <= 100, "Page size cannot exceed 100");
        
        if (startIndex >= records.length) {
            return new MediaRecord[](0); // Return empty array if out of bounds
        }
        
        uint256 endIndex = startIndex + pageSize;
        if (endIndex > records.length) {
            endIndex = records.length;
        }

        uint256 resultLength = endIndex - startIndex;
        MediaRecord[] memory result = new MediaRecord[](resultLength);

        for (uint256 i = 0; i < resultLength; i++) {
            result[i] = records[startIndex + i];
        }

        return result;
    }

    function totalRecords() public view returns (uint256) {
        return records.length;
    }
    
    // Getter for nextId (useful for frontend)
    function getNextId() public view returns (uint256) {
        return nextId;
    }
}