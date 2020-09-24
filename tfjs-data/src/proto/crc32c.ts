/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */

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
