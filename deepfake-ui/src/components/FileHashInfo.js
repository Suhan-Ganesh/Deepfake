import React from "react";
import { CONTRACT_ADDRESS } from "../config";

const FileHashInfo = ({ fileHash, isDeepfake, confidence, timestamp, transactionHash }) => {
  // Function to open Etherscan contract page with file hash and transaction hash search
  const openEtherscanWithSearch = () => {
    // Create a search URL that includes both file hash and transaction hash
    const searchParams = new URLSearchParams();
    if (fileHash) {
      searchParams.set('a', fileHash);
    }
    if (transactionHash && transactionHash.startsWith("0x")) {
      searchParams.set('tx', transactionHash);
    }
    
    const etherscanUrl = `https://sepolia.etherscan.io/address/${CONTRACT_ADDRESS}?${searchParams.toString()}`;
    window.open(etherscanUrl, "_blank");
  };
  
  // Function to open Etherscan transaction page (if transaction hash is available and valid)
  const openEtherscanTransaction = () => {
    if (transactionHash && transactionHash.startsWith("0x") && !transactionHash.includes("duplicate") && !transactionHash.includes("local")) {
      const etherscanUrl = `https://sepolia.etherscan.io/tx/${transactionHash}`;
      window.open(etherscanUrl, "_blank");
    } else {
      openEtherscanWithSearch();
    }
  };

  // Check if this is a duplicate file (mock transaction hash)
  const isDuplicateFile = transactionHash && (transactionHash.includes("duplicate") || transactionHash.includes("local"));

  return (
    <div className="mt-4 p-4 rounded-xl border-2 shadow-xl max-w-md w-full"
         style={{
           background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 100%)',
           borderColor: 'rgba(236, 72, 153, 0.5)',
           boxShadow: '0 10px 25px rgba(0, 0, 0, 0.3)'
         }}>
      <h2 className="text-xl font-bold text-white mb-4">File Hash Information</h2>
      
      <div className="space-y-3">
        {/* File Hash Section */}
        <div>
          <p className="text-indigo-300 text-sm font-semibold mb-1">File Hash</p>
          <p className="text-white font-mono text-xs break-all">{fileHash}</p>
        </div>
        
        <div className="border-t border-indigo-900"></div>
        
        {/* Detection Result Section */}
        <div>
          <p className="text-indigo-300 text-sm font-semibold mb-1">Detection Result</p>
          <p className={isDeepfake ? "text-red-500 font-bold text-lg" : "text-green-500 font-bold text-lg"}>
            {isDeepfake ? "DEEPFAKE" : "AUTHENTIC"}
          </p>
        </div>
        
        <div className="border-t border-indigo-900"></div>
        
        {/* Confidence Score Section */}
        <div>
          <p className="text-indigo-300 text-sm font-semibold mb-1">Confidence Score</p>
          <div className="flex items-center">
            <div className="w-full mr-2">
              <div className="w-full rounded-full h-2 bg-indigo-900 bg-opacity-50">
                <div 
                  className="h-2 rounded-full transition-all duration-500"
                  style={{
                    width: `${confidence * 100}%`,
                    background: isDeepfake ? 'linear-gradient(to right, #ef4444, #8b5cf6)' : 'linear-gradient(to right, #10b981, #8b5cf6)'
                  }}
                ></div>
              </div>
            </div>
            <span className="text-white text-sm min-w-[45px] text-right">
              {(confidence * 100).toFixed(2)}%
            </span>
          </div>
        </div>
        
        <div className="border-t border-indigo-900"></div>
        
        {/* Timestamp Section */}
        <div>
          <p className="text-indigo-300 text-sm font-semibold mb-1">Timestamp</p>
          <p className="text-white text-sm">{timestamp}</p>
        </div>
        
        <div className="border-t border-indigo-900"></div>
        
        {/* Blockchain Verification Section */}
        <div>
          <p className="text-indigo-300 text-sm font-semibold mb-2">Blockchain Verification</p>
          
          <button
            onClick={isDuplicateFile ? openEtherscanWithSearch : openEtherscanTransaction}
            className="w-full bg-gradient-to-r from-teal-500 to-cyan-600 hover:from-teal-600 hover:to-cyan-700 text-white py-2 rounded-lg transition font-semibold shadow-md hover:shadow-lg flex items-center justify-center space-x-2 mb-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>
              {isDuplicateFile 
                ? "Search File on Etherscan" 
                : "View Transaction on Etherscan"}
            </span>
          </button>
          
          <p className="text-gray-300 font-mono text-xs break-all mt-2">{CONTRACT_ADDRESS}</p>
          
          <div className="mt-3 p-2 rounded-lg bg-indigo-900 bg-opacity-30">
            <p className="text-xs text-indigo-200">
              <span className="font-semibold">Tip:</span> On the contract page, look for the "MediaRegistered" events to find your file record.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileHashInfo;