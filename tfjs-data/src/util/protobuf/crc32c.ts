import * as crc32c from 'fast-crc32c';

const kCrc32MaskDelta = 0xa282ead8;

const fourGb = Math.pow(2, 32);

// CRC-masking function used by TensorFlow.
function maskCrc(value: number): number {
  return (((value >>> 15) | (value << 17)) + kCrc32MaskDelta) % fourGb;
}

// Computes the masked CRC32C version used by TensorFlow.
export function maskedCrc32c(buffer: Buffer): number {
  const rawCrc: number = crc32c.calculate(buffer);
  return maskCrc(rawCrc);
}
