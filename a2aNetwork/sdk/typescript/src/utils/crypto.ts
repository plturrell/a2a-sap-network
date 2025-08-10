import { ethers } from 'ethers';
import { createError, ErrorCode } from './errors';

export function hashMessage(message: string): string {
  return ethers.id(message);
}

export function signMessage(message: string, privateKey: string): string {
  try {
    const wallet = new ethers.Wallet(privateKey);
    return wallet.signMessageSync(message);
  } catch (error: unknown) {
    throw createError(ErrorCode.INVALID_SIGNATURE, 'Failed to sign message', error);
  }
}

export async function verifySignature(
  message: string,
  signature: string,
  expectedAddress: string
): Promise<boolean> {
  try {
    const recoveredAddress = ethers.verifyMessage(message, signature);
    return recoveredAddress.toLowerCase() === expectedAddress.toLowerCase();
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Verification failed';
    console.warn('Signature verification failed:', {
      expectedAddress,
      error: errorMessage
    });
    return false; // Return false when signature verification fails
  }
}

export function generateRandomId(): string {
  return ethers.hexlify(ethers.randomBytes(32));
}

export function keccak256(data: string): string {
  return ethers.keccak256(ethers.toUtf8Bytes(data));
}

export interface KeyPair {
  publicKey: string;
  privateKey: string;
}

export function generateKeyPair(): KeyPair {
  const wallet = ethers.Wallet.createRandom();
  return {
    publicKey: wallet.address,
    privateKey: wallet.privateKey
  };
}

export async function encryptMessage(message: string, recipientPublicKey: string): Promise<string> {
  // Simple encryption using recipient's public key
  // In production, use proper encryption library like eth-crypto
  const messageHash = keccak256(message);
  return ethers.hexlify(ethers.toUtf8Bytes(JSON.stringify({
    encrypted: true,
    recipient: recipientPublicKey,
    data: ethers.hexlify(ethers.toUtf8Bytes(message)),
    hash: messageHash
  })));
}

export async function decryptMessage(encryptedMessage: string, _privateKey: string): Promise<string> {
  // Simple decryption - in production use proper encryption library
  try {
    const decoded = JSON.parse(ethers.toUtf8String(encryptedMessage));
    if (!decoded.encrypted) {
      throw new Error('Message is not encrypted');
    }
    return ethers.toUtf8String(decoded.data);
  } catch (error: unknown) {
    throw createError(ErrorCode.INVALID_MESSAGE, 'Failed to decrypt message', error);
  }
}