import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [chainData, setChainData] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploadResult(null); // Clear previous results
      
      const res = await axios.post("http://127.0.0.1:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setUploadResult(res.data);
      
      // Removed alert messages to prevent pop-ups
      // The results are displayed in the UI instead
      
    } catch (err) {
      console.error(err);
      alert("Upload failed! " + (err.response?.data?.error || err.message));
    }
  };

  const fetchChain = async () => {
    try {
      const res = await axios.get("http://127.0.0.1:5000/chain");

      setChainData(res.data);
    } catch (err) {
      console.error(err);
      alert("Failed to fetch chain!");
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-6">
      <h1 className="text-3xl font-bold mb-4">🔍 Deepfake Detection & Blockchain Verifier</h1>

      <div className="bg-gray-800 p-6 rounded-xl shadow-lg w-full max-w-md">
        <input
          type="file"
          accept="image/*,video/*"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-300 border border-gray-700 rounded-lg cursor-pointer bg-gray-900 mb-4 p-2"
        />

        <button
          onClick={handleUpload}
          className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-2 rounded-lg transition font-semibold"
        >
          🔍 Analyze & Upload to Blockchain
        </button>

        <button
          onClick={fetchChain}
          className="w-full bg-teal-600 hover:bg-teal-700 text-white py-2 rounded-lg mt-4 transition"
        >
          View Blockchain
        </button>
      </div>

      {uploadResult && (
        <div className="mt-6 p-4 bg-gray-800 rounded-lg w-full max-w-md border-2 shadow-xl"
             style={{
               borderColor: uploadResult.is_deepfake ? '#ef4444' : '#10b981'
             }}>
          <h2 className="text-xl font-semibold mb-3 flex items-center gap-2">
            {uploadResult.is_deepfake ? (
              <>
                <span className="text-2xl">⚠️</span>
                <span className="text-red-400">Deepfake Detected</span>
              </>
            ) : (
              <>
                <span className="text-2xl">✅</span>
                <span className="text-green-400">Authentic Media</span>
              </>
            )}
          </h2>
          
          <div className="space-y-2 text-sm">
            <p><b className="text-gray-400">Filename:</b> <span className="text-white">{uploadResult.filename}</span></p>
            <p><b className="text-gray-400">File Type:</b> <span className="text-white capitalize">{uploadResult.file_type}</span></p>
            <p><b className="text-gray-400">File Hash:</b> <span className="text-white font-mono text-xs break-all">{uploadResult.file_hash}</span></p>
            
            <div className="pt-2 border-t border-gray-700">
              <p className="mb-1"><b className="text-gray-400">Detection Confidence:</b></p>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="h-3 rounded-full transition-all"
                  style={{
                    width: `${uploadResult.confidence * 100}%`,
                    backgroundColor: uploadResult.is_deepfake ? '#ef4444' : '#10b981'
                  }}
                ></div>
              </div>
              <p className="text-xs text-gray-400 mt-1">
                {(uploadResult.confidence * 100).toFixed(2)}% 
                {uploadResult.is_deepfake ? ' likely deepfake' : ' likely authentic'}
              </p>
            </div>
            
            <p><b className="text-gray-400">Detection Method:</b> <span className="text-white">{uploadResult.detection_method}</span></p>
            <p><b className="text-gray-400">Transaction Hash:</b> <span className="text-white font-mono text-xs break-all">{uploadResult.transaction_hash}</span></p>
          </div>
        </div>
      )}

      {chainData && (
        <div className="mt-6 bg-gray-800 p-4 rounded-lg w-full max-w-3xl overflow-x-auto">
          <h2 className="text-xl font-semibold mb-2">Blockchain Data</h2>
          <pre className="text-sm bg-gray-900 p-2 rounded">{JSON.stringify(chainData, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;