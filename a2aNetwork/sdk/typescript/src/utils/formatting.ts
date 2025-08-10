import { ethers } from 'ethers';

export function formatAddress(address: string): string {
  return ethers.getAddress(address);
}

export function formatEther(wei: bigint | string): string {
  return ethers.formatEther(wei);
}

export function parseEther(ether: string): bigint {
  return ethers.parseEther(ether);
}

export function shortenAddress(address: string, chars = 4): string {
  const formatted = formatAddress(address);
  return `${formatted.slice(0, chars + 2)}...${formatted.slice(-chars)}`;
}

export function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toISOString();
}

export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  
  const parts = [];
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  if (secs > 0 || parts.length === 0) parts.push(`${secs}s`);
  
  return parts.join(' ');
}