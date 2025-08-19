'use client';

import React from 'react';
import { NextUIProvider } from '@nextui-org/react';
import { WagmiConfig } from 'wagmi';
import { RainbowKitProvider, darkTheme } from '@rainbow-me/rainbowkit';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { ThemeProvider as NextThemesProvider } from 'next-themes';

import { config, chains } from '../lib/wagmi';
import Navbar from '../components/layout/Navbar';
import Footer from '../components/layout/Footer';

import '@rainbow-me/rainbowkit/styles.css';
import './globals.css';

const queryClient = new QueryClient();

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <title>A2A Network - Autonomous Agent Ecosystem</title>
        <meta name="description" content="Decentralized marketplace for autonomous AI agents" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body className="font-a2a antialiased">
        <QueryClientProvider client={queryClient}>
          <WagmiConfig config={config}>
            <RainbowKitProvider
              chains={chains}
              theme={darkTheme({
                accentColor: '#0070f3',
                accentColorForeground: 'white',
                borderRadius: 'medium',
              })}
            >
              <NextThemesProvider attribute="class" defaultTheme="dark">
                <NextUIProvider>
                  <div className="min-h-screen bg-gradient-to-br from-a2a-dark via-slate-900 to-a2a-dark">
                    <Navbar />
                    <main className="flex-1">
                      {children}
                    </main>
                    <Footer />
                  </div>
                  <Toaster
                    position="bottom-right"
                    toastOptions={{
                      style: {
                        background: '#1f2937',
                        color: '#f3f4f6',
                        border: '1px solid #374151',
                      },
                    }}
                  />
                </NextUIProvider>
              </NextThemesProvider>
            </RainbowKitProvider>
          </WagmiConfig>
        </QueryClientProvider>
      </body>
    </html>
  );
}