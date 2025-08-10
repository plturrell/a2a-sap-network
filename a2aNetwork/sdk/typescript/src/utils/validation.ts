import { ethers } from 'ethers';
import { ErrorCode, createError } from './errors';

export function isValidAddress(address: string): boolean {
  return ethers.isAddress(address);
}

export function validateAddress(address: string): void {
  if (!isValidAddress(address)) {
    throw createError(ErrorCode.INVALID_ADDRESS, `Invalid Ethereum address: ${address}`);
  }
}

export function isValidPrivateKey(privateKey: string): boolean {
  try {
    new ethers.Wallet(privateKey);
    return true;
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Invalid format';
    console.warn('Invalid private key format:', {
      error: errorMessage,
      keyLength: privateKey?.length || 0
    });
    return false; // Return false for invalid private keys
  }
}

export function validatePrivateKey(privateKey: string): void {
  if (!isValidPrivateKey(privateKey)) {
    throw createError(ErrorCode.VALIDATION_ERROR, 'Invalid private key format');
  }
}

export function validateRequiredString(value: string | undefined, fieldName: string): void {
  if (!value || value.trim().length === 0) {
    throw createError(ErrorCode.VALIDATION_ERROR, `${fieldName} is required`);
  }
}

export function validatePositiveNumber(value: number | undefined, fieldName: string): void {
  if (value === undefined || value <= 0) {
    throw createError(ErrorCode.VALIDATION_ERROR, `${fieldName} must be a positive number`);
  }
}

export function validateAgentParams(params: {
  name?: string;
  capabilities?: unknown;
  [key: string]: unknown;
}): { isValid: boolean; errors?: string[] } {
  const errors: string[] = [];
  
  if (!params.name || params.name.trim().length === 0) {
    errors.push('Agent name is required');
  }
  
  if (!params.capabilities || !Array.isArray(params.capabilities) || params.capabilities.length === 0) {
    errors.push('At least one capability is required');
  }
  
  return {
    isValid: errors.length === 0,
    errors: errors.length > 0 ? errors : undefined
  };
}

export function validateConfig(config: {
  network?: string;
  rpcUrl?: string;
  provider?: unknown;
  privateKey?: string;
  [key: string]: unknown;
}): { isValid: boolean; errors?: string[] } {
  const errors: string[] = [];
  
  if (!config.network) {
    errors.push('Network is required');
  }
  
  if (!config.rpcUrl && !config.provider) {
    errors.push('Either rpcUrl or provider is required');
  }
  
  if (config.privateKey && !isValidPrivateKey(config.privateKey)) {
    errors.push('Invalid private key format');
  }
  
  return {
    isValid: errors.length === 0,
    errors: errors.length > 0 ? errors : undefined
  };
}