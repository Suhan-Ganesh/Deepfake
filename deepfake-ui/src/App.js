import React, { useState } from "react";
import axios from "axios";
import SplashScreen from "./components/SplashScreen";
import AuthScreen from "./components/AuthScreen";
import Logo from "./components/Logo";
import BlockchainRecords from "./components/BlockchainRecords";
import BlockchainViewer from "./components/BlockchainViewer";

function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authData, setAuthData] = useState(null);
  const [file, setFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [chainData, setChainData] = useState(null);
  const [showProfilePopup, setShowProfilePopup] = useState(false);
  const [showBlockchainInfo, setShowBlockchainInfo] = useState(false);

  const handleSplashFinish = () => {
    setShowSplash(false);
  };

  const handleAuthSuccess = (data) => {
    setAuthData(data);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setAuthData(null);
    setFile(null);
    setUploadResult(null);
    setChainData(null);
    setShowBlockchainInfo(false);
  };

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
      setChainData(null); // Hide blockchain records when uploading new file
      setShowBlockchainInfo(false); // Hide blockchain info when uploading new file
      
      const res = await axios.post("http://127.0.0.1:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // If we have a transaction hash, check its status
      if (res.data.transaction_hash && !res.data.duplicate) {
        try {
          const statusRes = await axios.get(`http://127.0.0.1:5000/transaction-status/${res.data.transaction_hash}`);
          console.log("Transaction status:", statusRes.data);
        } catch (statusErr) {
          console.error("Error checking transaction status:", statusErr);
        }
      }

      setUploadResult(res.data);
      
      // Removed alert messages to prevent pop-ups
      // The results are displayed in the UI instead
      
    } catch (err) {
      console.error(err);
      alert("Upload failed! " + (err.response?.data?.error || err.message));
    }
  };
      
  // Show splash screen
  if (showSplash) {
    return <SplashScreen onFinish={handleSplashFinish} />;
  }

  // Show authentication screen
  if (!isAuthenticated) {
    return <AuthScreen onAuthSuccess={handleAuthSuccess} />;
  }

  const fetchChain = async () => {
    try {
      // Simply hide the blockchain records when clicking the button
      setChainData(null);
      setShowBlockchainInfo(!showBlockchainInfo);
    } catch (err) {
      console.error(err);
      alert("Failed to fetch chain!");
    }
  };

  return (
    <div className="min-h-screen flex flex-col p-6 relative overflow-hidden" style={{
      background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 25%, #4c1d95 50%, #581c87 75%, #3b0764 100%)'
    }}>
      {/* Modern Abstract Background Elements */}
      <div className="wave-shape wave-1" style={{ opacity: 0.25 }}></div>
      <div className="wave-shape wave-2" style={{ opacity: 0.25 }}></div>
      <div className="wave-shape wave-3" style={{ opacity: 0.25 }}></div>
      <div className="glowing-dots" style={{ opacity: 0.15 }}></div>
      <div className="diagonal-lines" style={{ opacity: 0.1 }}></div>
      <div className="translucent-circle circle-1" style={{ opacity: 0.2 }}></div>
      <div className="translucent-circle circle-2" style={{ opacity: 0.2 }}></div>
      <div className="translucent-circle circle-3" style={{ opacity: 0.2 }}></div>
      <div className="lighting-effect" style={{ opacity: 0.15 }}></div>
      
      {/* Floating Glowing Orbs Background */}
      <div className="glow-orb glow-orb-1" style={{ background: 'radial-gradient(circle, rgba(139, 92, 246, 0.4), transparent)' }}></div>
      <div className="glow-orb glow-orb-2" style={{ background: 'radial-gradient(circle, rgba(59, 130, 246, 0.3), transparent)' }}></div>
      <div className="glow-orb glow-orb-3" style={{ background: 'radial-gradient(circle, rgba(168, 85, 247, 0.4), transparent)' }}></div>
      <div className="glow-orb glow-orb-4" style={{ background: 'radial-gradient(circle, rgba(96, 165, 250, 0.3), transparent)' }}></div>
      <div className="glow-orb glow-orb-5" style={{ background: 'radial-gradient(circle, rgba(147, 51, 234, 0.4), transparent)' }}></div>
      <div className="glow-orb glow-orb-6" style={{ background: 'radial-gradient(circle, rgba(124, 58, 237, 0.35), transparent)' }}></div>
      
      {/* Content Wrapper with higher z-index */}
      <div className="relative z-10 flex flex-col flex-1">
      {/* Profile Popup */}
      {showProfilePopup && (
        <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50 fade-in" onClick={() => setShowProfilePopup(false)}>
          <div className="rounded-2xl p-8 w-full max-w-md shadow-2xl" style={{
            background: 'rgba(30, 27, 75, 0.95)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(139, 92, 246, 0.3)'
          }} onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-white">Profile</h2>
              <button onClick={() => setShowProfilePopup(false)} className="text-indigo-300 hover:text-white transition">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            {/* User Info */}
            <div className="flex items-center space-x-4 mb-6 p-4 rounded-xl shadow-sm" style={{
              background: 'rgba(99, 102, 241, 0.15)',
              border: '1px solid rgba(139, 92, 246, 0.3)'
            }}>
              <img
                src={authData.user.picture}
                alt={authData.user.name}
                className="w-20 h-20 rounded-full border-4 border-indigo-400 shadow-lg"
              />
              <div>
                <h3 className="text-xl font-semibold text-white">{authData.user.name}</h3>
                <p className="text-sm text-indigo-200">{authData.user.email}</p>
              </div>
            </div>
            
            {/* Wallet Info */}
            <div className="rounded-xl p-4 mb-4 shadow-sm" style={{
              background: 'rgba(59, 130, 246, 0.15)',
              border: '1px solid rgba(96, 165, 250, 0.3)'
            }}>
              <h4 className="text-sm font-semibold text-indigo-200 mb-2 flex items-center">
                <svg className="w-4 h-4 mr-2 text-indigo-400" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M4 4a2 2 0 00-2 2v1h16V6a2 2 0 00-2-2H4z"/>
                  <path fillRule="evenodd" d="M18 9H2v5a2 2 0 002 2h12a2 2 0 002-2V9zM4 13a1 1 0 100 2h1a1 1 0 100-2H5a1 1 0 01-1-1zm5-1a1 1 0 100 2h1a1 1 0 100-2H9z" clipRule="evenodd"/>
                </svg>
                Connected Wallet
              </h4>
              <div className="flex items-center justify-between">
                <p className="text-sm font-mono text-white break-all">{authData.walletAddress}</p>
                <button 
                  onClick={() => {
                    navigator.clipboard.writeText(authData.walletAddress);
                    alert('Wallet address copied!');
                  }}
                  className="ml-2 p-2 hover:bg-indigo-800 rounded-lg transition"
                  title="Copy address"
                >
                  <svg className="w-5 h-5 text-indigo-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </button>
              </div>
            </div>
            
            {/* Account Stats */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="p-4 rounded-xl shadow-sm" style={{
                background: 'rgba(99, 102, 241, 0.15)',
                border: '1px solid rgba(139, 92, 246, 0.3)'
              }}>
                <p className="text-xs text-indigo-300 mb-1">Account Type</p>
                <p className="text-lg font-semibold text-indigo-400">Standard</p>
              </div>
              <div className="p-4 rounded-xl shadow-sm" style={{
                background: 'rgba(34, 197, 94, 0.15)',
                border: '1px solid rgba(74, 222, 128, 0.3)'
              }}>
                <p className="text-xs text-emerald-300 mb-1">Status</p>
                <p className="text-lg font-semibold text-emerald-400 flex items-center">
                  <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
                  Active
                </p>
              </div>
            </div>
            
            {/* Logout Button */}
            <button
              onClick={handleLogout}
              className="w-full bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white py-3 rounded-lg font-semibold transition shadow-md hover:shadow-lg flex items-center justify-center space-x-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
              <span>Logout</span>
            </button>
          </div>
        </div>
      )}
      
      {/* Header with Logo and User Info */}
      <header className="rounded-2xl shadow-lg p-4 mb-6 flex items-center justify-between fade-in" style={{
        background: 'rgba(30, 27, 75, 0.8)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(139, 92, 246, 0.3)'
      }}>
        <div className="flex items-center space-x-3">
          <Logo size={48} />
          <div>
            <h1 className="text-2xl font-bold text-white drop-shadow">Secura</h1>
            <p className="text-sm text-indigo-200">Deepfake Detection & Blockchain Verification</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* User Info - Clickable */}
          <button
            onClick={() => setShowProfilePopup(true)}
            className="flex items-center space-x-3 hover:bg-indigo-900 hover:bg-opacity-40 p-2 rounded-lg transition-all duration-200"
          >
            <img
              src={authData.user.picture}
              alt={authData.user.name}
              className="w-10 h-10 rounded-full border-2 border-indigo-400 shadow-md"
            />
            <div className="hidden md:block text-left">
              <p className="text-sm font-semibold text-white">{authData.user.name}</p>
              <p className="text-xs text-indigo-200">{authData.user.email}</p>
            </div>
          </button>
        </div>
      </header>
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col items-center justify-center">

        <div className="p-6 rounded-2xl shadow-lg w-full max-w-md fade-in" style={{
          background: 'rgba(30, 27, 75, 0.8)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(139, 92, 246, 0.3)'
        }}>
          <input
            type="file"
            accept="image/*,video/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-white border-2 rounded-lg cursor-pointer mb-4 p-2 transition-all file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-indigo-600 file:text-white hover:file:bg-indigo-700 file:cursor-pointer file:font-semibold"
            style={{
              background: 'rgba(99, 102, 241, 0.1)',
              borderColor: 'rgba(139, 92, 246, 0.4)'
            }}
          />

          <button
            onClick={handleUpload}
            className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white py-3 rounded-lg transition font-semibold shadow-md hover:shadow-lg pulse-glow flex items-center justify-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <span>🔍 Analyze & Upload to Blockchain</span>
          </button>

          <button
            onClick={fetchChain}
            className="w-full bg-gradient-to-r from-teal-500 to-cyan-600 hover:from-teal-600 hover:to-cyan-700 text-white py-3 rounded-lg mt-4 transition shadow-md hover:shadow-lg flex items-center justify-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span>{showBlockchainInfo ? "Hide Blockchain" : "View Blockchain"}</span>
          </button>
        </div>

      {uploadResult && (
        <div className="mt-6 p-4 rounded-xl w-full max-w-md shadow-xl fade-in"
             style={{
               background: 'rgba(30, 27, 75, 0.8)',
               backdropFilter: 'blur(20px)',
               border: '1px solid rgba(255, 165, 0, 0.5)',
               boxShadow: '0 0 15px rgba(255, 165, 0, 0.3)'
             }}>
          <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
            {uploadResult.duplicate ? (
              <>
                <span className="text-2xl">ℹ️</span>
                <span className="text-yellow-400">Duplicate File Detected</span>
              </>
            ) : uploadResult.is_deepfake ? (
              <>
                <span className="text-2xl">⚠️</span>
                <span className="text-orange-500">Deepfake Detected</span>
              </>
            ) : (
              <>
                <span className="text-2xl">✅</span>
                <span className="text-green-400">Authentic Media</span>
              </>
            )}
          </h2>
          
          <div className="space-y-3 text-sm">
            <div>
              <p className="text-indigo-300 font-semibold">Filename</p>
              <p className="text-white mt-1">{uploadResult.filename}</p>
            </div>
            
            <div>
              <p className="text-indigo-300 font-semibold">File Type</p>
              <p className="text-white capitalize mt-1">{uploadResult.file_type}</p>
            </div>
            
            <div>
              <p className="text-indigo-300 font-semibold">File Hash</p>
              <p className="text-blue-200 font-mono text-xs break-all mt-1">{uploadResult.file_hash}</p>
            </div>
            
            {uploadResult.duplicate ? (
              <div className="pt-2 border-t border-indigo-800">
                <p className="text-yellow-300 font-semibold">{uploadResult.duplicate_message}</p>
                <p className="text-indigo-300 text-xs mt-1">The file has already been analyzed and stored in the blockchain.</p>
              </div>
            ) : (
              <>
                <div className="pt-2 border-t border-indigo-800">
                  <div className="flex justify-between items-center mb-1">
                    <p className="text-white font-semibold">Detection Confidence</p>
                    <p className={uploadResult.is_deepfake ? "text-red-500 font-semibold" : "text-green-500 font-semibold"}>
                      {(uploadResult.confidence * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="w-full rounded-full h-2.5" style={{ background: 'rgba(99, 102, 241, 0.2)' }}>
                    <div 
                      className="h-2.5 rounded-full transition-all"
                      style={{
                        width: `${uploadResult.confidence * 100}%`,
                        backgroundColor: uploadResult.is_deepfake ? '#ef4444' : '#10b981'
                      }}
                    ></div>
                  </div>
                  <p className="text-indigo-300 text-xs mt-1">
                    {uploadResult.is_deepfake ? 'Likely deepfake' : 'Likely authentic'}
                  </p>
                </div>
                
                <div>
                  <p className="text-indigo-300 font-semibold">Detection Method</p>
                  <p className="text-white mt-1">{uploadResult.detection_method}</p>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {showBlockchainInfo && uploadResult && !uploadResult.duplicate && uploadResult.transaction_hash && (
        <BlockchainViewer 
          fileHash={uploadResult.file_hash}
          isDeepfake={uploadResult.is_deepfake}
          confidence={uploadResult.confidence}
          transactionHash={uploadResult.transaction_hash}
        />
      )}

      {chainData && (
        <BlockchainRecords records={chainData.chain} />
      )}
    </div>
  </div>
</div>
  );
}

export default App;
