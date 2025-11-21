import React, { useEffect, useState } from 'react';
import Logo from './Logo';

const SplashScreen = ({ onFinish }) => {
  const [fadeOut, setFadeOut] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setFadeOut(true);
      setTimeout(onFinish, 500); // Wait for fade out animation
    }, 2500);

    return () => clearTimeout(timer);
  }, [onFinish]);

  return (
    <div
      className={`fixed inset-0 flex flex-col items-center justify-center z-50 transition-opacity duration-500 overflow-hidden ${
        fadeOut ? 'opacity-0' : 'opacity-100'
      }`}
      style={{
        background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 25%, #4c1d95 50%, #581c87 75%, #3b0764 100%)'
      }}
    >
      {/* Flowing Wave Shapes - Same as light theme */}
      <div className="wave-shape wave-1" style={{ opacity: 0.25 }}></div>
      <div className="wave-shape wave-2" style={{ opacity: 0.25 }}></div>
      <div className="wave-shape wave-3" style={{ opacity: 0.25 }}></div>
      
      {/* Glowing Dots Pattern */}
      <div className="glowing-dots" style={{ opacity: 0.15 }}></div>
      
      {/* Diagonal Lines */}
      <div className="diagonal-lines" style={{ opacity: 0.1 }}></div>
      
      {/* Translucent Circles */}
      <div className="translucent-circle circle-1" style={{ opacity: 0.2 }}></div>
      <div className="translucent-circle circle-2" style={{ opacity: 0.2 }}></div>
      
      {/* Lighting Effect */}
      <div className="lighting-effect" style={{ opacity: 0.15 }}></div>
      
      {/* Floating Glowing Orbs - Same as light theme */}
      <div className="glow-orb glow-orb-1" style={{ background: 'radial-gradient(circle, rgba(139, 92, 246, 0.4), transparent)' }}></div>
      <div className="glow-orb glow-orb-2" style={{ background: 'radial-gradient(circle, rgba(59, 130, 246, 0.3), transparent)' }}></div>
      <div className="glow-orb glow-orb-3" style={{ background: 'radial-gradient(circle, rgba(168, 85, 247, 0.4), transparent)' }}></div>
      <div className="glow-orb glow-orb-4" style={{ background: 'radial-gradient(circle, rgba(96, 165, 250, 0.3), transparent)' }}></div>
      <div className="glow-orb glow-orb-5" style={{ background: 'radial-gradient(circle, rgba(147, 51, 234, 0.4), transparent)' }}></div>
      
      {/* Main Content */}
      <div className="relative z-10 flex flex-col items-center">
        <div className="float mb-8">
          <Logo size={140} />
        </div>
        
        <h1 className="text-6xl font-bold text-white mb-4 animate-pulse drop-shadow-2xl" style={{ textShadow: '0 0 40px rgba(139, 92, 246, 0.8)' }}>
          Secura
        </h1>
        
        <p className="text-xl text-indigo-200 mb-10 drop-shadow-lg font-medium">
          Deepfake Detection & Blockchain Verification
        </p>
        
        {/* Loading spinner with glow effect */}
        <div className="flex space-x-3">
          <div className="w-4 h-4 bg-indigo-400 rounded-full animate-bounce shadow-lg" style={{ animationDelay: '0s', boxShadow: '0 0 20px rgba(139, 92, 246, 0.8)' }}></div>
          <div className="w-4 h-4 bg-purple-400 rounded-full animate-bounce shadow-lg" style={{ animationDelay: '0.2s', boxShadow: '0 0 20px rgba(168, 85, 247, 0.8)' }}></div>
          <div className="w-4 h-4 bg-blue-400 rounded-full animate-bounce shadow-lg" style={{ animationDelay: '0.4s', boxShadow: '0 0 20px rgba(59, 130, 246, 0.8)' }}></div>
        </div>
        
        {/* Loading text */}
        <p className="mt-6 text-indigo-300 text-sm font-medium animate-pulse">Loading...</p>
      </div>
    </div>
  );
};

export default SplashScreen;
