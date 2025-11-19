// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DeepfakeAuth {
    struct MediaRecord {
        string fileHash;
        address uploader;
        uint256 timestamp;
    }

    MediaRecord[] public records;

    event MediaRegistered(string fileHash, address uploader, uint256 timestamp);

    function registerMedia(string memory _fileHash) public {
        records.push(MediaRecord(_fileHash, msg.sender, block.timestamp));
        emit MediaRegistered(_fileHash, msg.sender, block.timestamp);
    }

    function getAllRecords() public view returns (MediaRecord[] memory) {
        return records;
    }

    function totalRecords() public view returns (uint256) {
        return records.length;
    }
}
