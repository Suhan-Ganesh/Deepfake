import React from 'react';

const Logo = ({ size = 48, className = "" }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      className={className}
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Shield background */}
      <path
        d="M50 5 L85 20 L85 50 Q85 75 50 95 Q15 75 15 50 L15 20 Z"
        fill="url(#gradient1)"
        stroke="#6366f1"
        strokeWidth="2"
      />
      
      {/* Checkmark for authenticity */}
      <path
        d="M35 50 L45 60 L65 35"
        stroke="#10b981"
        strokeWidth="4"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      
      {/* Chain links */}
      <circle cx="30" cy="70" r="6" fill="none" stroke="#8b5cf6" strokeWidth="2.5" />
      <circle cx="45" cy="70" r="6" fill="none" stroke="#8b5cf6" strokeWidth="2.5" />
      <circle cx="55" cy="70" r="6" fill="none" stroke="#8b5cf6" strokeWidth="2.5" />
      <circle cx="70" cy="70" r="6" fill="none" stroke="#8b5cf6" strokeWidth="2.5" />
      
      {/* Gradient definition */}
      <defs>
        <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#4f46e5" stopOpacity="0.8" />
          <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.9" />
        </linearGradient>
      </defs>
    </svg>
  );
};

export default Logo;
