// Minimal typings for fast-crc32c

declare module 'fast-crc32c' {
  export function calculate(input: Buffer): number;
}
