const { nextui } = require('@nextui-org/react');

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './node_modules/@nextui-org/theme/dist/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'a2a-primary': '#0070f3',
        'a2a-secondary': '#7928ca',
        'a2a-success': '#10b981',
        'a2a-warning': '#f59e0b',
        'a2a-error': '#ef4444',
        'a2a-dark': '#0f0f23',
        'a2a-light': '#f8fafc',
      },
      fontFamily: {
        'a2a': ['Inter', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-slow': 'pulse 3s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'a2a-gradient': 'linear-gradient(135deg, #0070f3 0%, #7928ca 100%)',
      },
    },
  },
  darkMode: 'class',
  plugins: [
    nextui({
      themes: {
        light: {
          colors: {
            primary: '#0070f3',
            secondary: '#7928ca',
            success: '#10b981',
            warning: '#f59e0b',
            danger: '#ef4444',
          },
        },
        dark: {
          colors: {
            primary: '#0070f3',
            secondary: '#7928ca',
            success: '#10b981',
            warning: '#f59e0b',
            danger: '#ef4444',
            background: '#0f0f23',
          },
        },
      },
    }),
  ],
};