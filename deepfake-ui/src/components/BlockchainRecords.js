import React from "react";

const BlockchainRecords = ({ records }) => {
  if (!records || records.length === 0) {
    return (
      <div className="mt-4 p-4 rounded-xl text-center" style={{
        background: 'rgba(30, 27, 75, 0.8)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(139, 92, 246, 0.3)'
      }}>
        <p className="text-indigo-200">No blockchain records found</p>
      </div>
    );
  }

  return (
    <div className="mt-6 w-full max-w-3xl overflow-x-auto shadow-lg fade-in" style={{
      background: 'rgba(30, 27, 75, 0.8)',
      backdropFilter: 'blur(20px)',
      border: '1px solid rgba(139, 92, 246, 0.3)'
    }}>
      <h2 className="text-xl font-semibold mb-4 text-white px-4 pt-4">Blockchain Records ({records.length})</h2>
      <div className="overflow-y-auto max-h-[500px] px-4 pb-4">
        {records.map((record, index) => (
          <div 
            key={index} 
            className="mb-4 p-4 rounded-lg border"
            style={{
              background: 'rgba(99, 102, 241, 0.1)',
              borderColor: record.isDeepfake ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)'
            }}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <p className="text-xs text-indigo-300 mb-1">File Hash</p>
                <p className="text-white font-mono text-xs break-all">{record.fileHash}</p>
              </div>
              <div>
                <p className="text-xs text-indigo-300 mb-1">Uploader Address</p>
                <p className="text-white font-mono text-xs break-all">{record.uploader}</p>
              </div>
              <div>
                <p className="text-xs text-indigo-300 mb-1">Detection Result</p>
                <p className={record.isDeepfake ? "text-red-400 font-semibold" : "text-green-400 font-semibold"}>
                  {record.isDeepfake ? "DEEPFAKE" : "AUTHENTIC"}
                </p>
              </div>
              <div>
                <p className="text-xs text-indigo-300 mb-1">Confidence Score</p>
                <div className="flex items-center">
                  <div className="w-full mr-2">
                    <div className="w-full rounded-full h-2" style={{ background: 'rgba(99, 102, 241, 0.2)' }}>
                      <div 
                        className="h-2 rounded-full transition-all"
                        style={{
                          width: `${record.confidenceScore * 100}%`,
                          backgroundColor: record.isDeepfake ? '#ef4444' : '#10b981'
                        }}
                      ></div>
                    </div>
                  </div>
                  <span className="text-xs text-white min-w-[40px]">
                    {(record.confidenceScore * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
              <div>
                <p className="text-xs text-indigo-300 mb-1">Timestamp</p>
                <p className="text-white text-sm">
                  {new Date(record.timestamp * 1000).toLocaleString()}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BlockchainRecords;