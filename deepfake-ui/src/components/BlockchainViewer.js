import React, { useState } from "react";
import FileHashInfo from "./FileHashInfo";

const BlockchainViewer = ({ fileHash, isDeepfake, confidence }) => {
  return (
    <FileHashInfo 
      fileHash={fileHash}
      isDeepfake={isDeepfake}
      confidence={confidence}
      timestamp={new Date().toLocaleString()}
    />
  );
};

export default BlockchainViewer;