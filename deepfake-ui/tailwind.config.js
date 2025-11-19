/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}", // look for Tailwind classes in src
  ],
  theme: {
    extend: {
      colors: {
        primary: "#1E293B",  // subtle dark background
        accent: "#38bdf8",   // cyan accent
      },
    },
  },
  plugins: [],
}
