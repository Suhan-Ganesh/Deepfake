import React from "react";
import FileHashInfo from "./FileHashInfo";

const BlockchainViewer = ({ fileHash, isDeepfake, confidence, transactionHash }) => {
  return (
    <FileHashInfo 
      fileHash={fileHash}
      isDeepfake={isDeepfake}
      confidence={confidence}
      timestamp={new Date().toLocaleString()}
      transactionHash={transactionHash}
    />
  );
};

export default BlockchainViewer;