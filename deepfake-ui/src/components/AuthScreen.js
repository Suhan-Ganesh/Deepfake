import React, { useState } from 'react';
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';
import { jwtDecode } from 'jwt-decode';
import Logo from './Logo';

const AuthScreen = ({ onAuthSuccess }) => {
  const [user, setUser] = useState(null);
  const [walletAddress, setWalletAddress] = useState(null);
  const [isConnectingWallet, setIsConnectingWallet] = useState(false);

  // Replace with your Google OAuth Client ID
  // Get it from: https://console.cloud.google.com/apis/credentials
  const GOOGLE_CLIENT_ID = "743049544920-4aih8jq86pq3es9htormp9lua997m5pi.apps.googleusercontent.com";

  const handleGoogleSuccess = (credentialResponse) => {
    try {
      const decoded = jwtDecode(credentialResponse.credential);
      setUser({
        name: decoded.name,
        email: decoded.email,
        picture: decoded.picture,
      });
    } catch (error) {
      console.error('Google login error:', error);
    }
  };

  const handleGoogleError = () => {
    console.error('Google Login Failed');
    alert('Google login failed. Please try again.');
  };

  const connectWallet = async () => {
    setIsConnectingWallet(true);
    
    try {
      console.log('Checking for MetaMask...', window.ethereum);
      
      if (!window.ethereum) {
        alert('MetaMask is not installed! Please install MetaMask extension and refresh the page.');
        setIsConnectingWallet(false);
        return;
      }

      console.log('MetaMask found! Requesting accounts...');
      
      // Request account access
      const accounts = await window.ethereum.request({
        method: 'eth_requestAccounts',
      });

      console.log('Accounts received:', accounts);
      setWalletAddress(accounts[0]);
      
      // Listen for account changes
      window.ethereum.on('accountsChanged', (accounts) => {
        if (accounts.length === 0) {
          setWalletAddress(null);
        } else {
          setWalletAddress(accounts[0]);
        }
      });
      
    } catch (error) {
      console.error('Wallet connection error:', error);
      alert('Failed to connect wallet. Please try again.');
    } finally {
      setIsConnectingWallet(false);
    }
  };

  const handleContinue = () => {
    if (user && walletAddress) {
      onAuthSuccess({ user, walletAddress });
    }
  };

  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <div className="min-h-screen flex items-center justify-center p-6 relative overflow-hidden" style={{
        background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 25%, #4c1d95 50%, #581c87 75%, #3b0764 100%)'
      }}>
        {/* Modern Abstract Background */}
        <div className="wave-shape wave-1" style={{ opacity: 0.25 }}></div>
        <div className="wave-shape wave-2" style={{ opacity: 0.25 }}></div>
        <div className="glowing-dots" style={{ opacity: 0.15 }}></div>
        <div className="diagonal-lines" style={{ opacity: 0.1 }}></div>
        <div className="translucent-circle circle-1" style={{ opacity: 0.2 }}></div>
        <div className="translucent-circle circle-2" style={{ opacity: 0.2 }}></div>
        <div className="lighting-effect" style={{ opacity: 0.15 }}></div>
        
        {/* Floating Glowing Orbs */}
        <div className="glow-orb glow-orb-1" style={{ background: 'radial-gradient(circle, rgba(139, 92, 246, 0.4), transparent)' }}></div>
        <div className="glow-orb glow-orb-2" style={{ background: 'radial-gradient(circle, rgba(59, 130, 246, 0.3), transparent)' }}></div>
        <div className="glow-orb glow-orb-3" style={{ background: 'radial-gradient(circle, rgba(168, 85, 247, 0.4), transparent)' }}></div>
        <div className="glow-orb glow-orb-4" style={{ background: 'radial-gradient(circle, rgba(96, 165, 250, 0.3), transparent)' }}></div>
        
        <div className="rounded-2xl shadow-2xl p-8 w-full max-w-md fade-in relative z-10" style={{
          background: 'rgba(30, 27, 75, 0.9)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(139, 92, 246, 0.3)'
        }}>
          {/* Logo and Title */}
          <div className="flex flex-col items-center mb-8">
            <Logo size={80} className="mb-4" />
            <h1 className="text-3xl font-bold text-white mb-2 drop-shadow-lg">Welcome to Secura</h1>
            <p className="text-indigo-200 text-center">
              Authenticate and connect your wallet to continue
            </p>
          </div>

          {/* Google Authentication */}
          <div className="mb-6">
            <h3 className="text-white font-semibold mb-3 flex items-center">
              <span className="text-2xl mr-2">1️⃣</span>
              Sign in with Google
            </h3>
            
            {!user ? (
              <div className="flex justify-center">
                <GoogleLogin
                  onSuccess={handleGoogleSuccess}
                  onError={handleGoogleError}
                  useOneTap
                  theme="filled_black"
                  size="large"
                  width="350"
                />
              </div>
            ) : (
              <div className="rounded-lg p-4 flex items-center space-x-3 shadow-sm" style={{
                background: 'rgba(34, 197, 94, 0.15)',
                border: '1px solid rgba(74, 222, 128, 0.3)'
              }}>
                <img
                  src={user.picture}
                  alt={user.name}
                  className="w-12 h-12 rounded-full border-2 border-green-400 shadow"
                />
                <div className="flex-1">
                  <p className="text-white font-semibold">{user.name}</p>
                  <p className="text-emerald-200 text-sm">{user.email}</p>
                </div>
                <button
                  onClick={() => setUser(null)}
                  className="text-red-400 hover:text-red-300 text-sm font-medium"
                >
                  Change
                </button>
              </div>
            )}
          </div>

          {/* Wallet Connection */}
          <div className="mb-6">
            <h3 className="text-white font-semibold mb-3 flex items-center">
              <span className="text-2xl mr-2">2️⃣</span>
              Connect Wallet
            </h3>
            
            {!walletAddress ? (
              <button
                onClick={connectWallet}
                disabled={!user || isConnectingWallet}
                className={`w-full py-3 rounded-lg font-semibold transition flex items-center justify-center space-x-2 shadow-md ${
                  user && !isConnectingWallet
                    ? 'bg-orange-500 hover:bg-orange-600 text-white hover:shadow-lg'
                    : 'text-gray-500 cursor-not-allowed'
                }`}
                style={{
                  background: user && !isConnectingWallet ? undefined : 'rgba(99, 102, 241, 0.1)',
                  border: user && !isConnectingWallet ? undefined : '1px solid rgba(139, 92, 246, 0.3)'
                }}
              >
                <svg className="w-6 h-6" viewBox="0 0 40 40" fill="none">
                  <path d="M32.5 18.75V11.25L20 3.75L7.5 11.25V18.75L20 26.25L32.5 18.75Z" fill="#E17726" stroke="#E17726" strokeWidth="0.5"/>
                  <path d="M20 26.25V36.25L32.5 18.75L20 26.25Z" fill="#E27625" stroke="#E27625" strokeWidth="0.5"/>
                  <path d="M20 26.25V36.25L7.5 18.75L20 26.25Z" fill="#E27625" stroke="#E27625" strokeWidth="0.5"/>
                </svg>
                <span>{isConnectingWallet ? 'Connecting...' : 'Connect MetaMask'}</span>
              </button>
            ) : (
              <div className="rounded-lg p-4 shadow-sm" style={{
                background: 'rgba(59, 130, 246, 0.15)',
                border: '1px solid rgba(96, 165, 250, 0.3)'
              }}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-indigo-300 text-xs mb-1">Connected Wallet</p>
                    <p className="text-white font-mono text-sm">
                      {walletAddress.slice(0, 6)}...{walletAddress.slice(-4)}
                    </p>
                  </div>
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse shadow-lg"></div>
                </div>
              </div>
            )}
          </div>

          {/* Continue Button */}
          <button
            onClick={handleContinue}
            disabled={!user || !walletAddress}
            className={`w-full py-3 rounded-lg font-semibold transition shadow-md hover:shadow-lg ${
              user && walletAddress
                ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
                : 'text-gray-500 cursor-not-allowed'
            }`}
            style={{
              background: user && walletAddress ? undefined : 'rgba(99, 102, 241, 0.1)',
              border: user && walletAddress ? undefined : '1px solid rgba(139, 92, 246, 0.3)'
            }}
          >
            Continue to App
          </button>

          <p className="text-indigo-300 text-xs text-center mt-4">
            By continuing, you agree to our Terms of Service
          </p>
        </div>
      </div>
    </GoogleOAuthProvider>
  );
};

export default AuthScreen;
