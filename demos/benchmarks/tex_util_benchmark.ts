/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
 * =============================================================================
 */

import * as tex_util from '../../src/math/webgl/tex_util';
import * as test_util from '../../src/test_util';

import {webgl_util} from '../deeplearn';

import {BenchmarkTest} from './benchmark';

const OPS_PER_RUN = 100;

export const BENCHMARK_ENCODE_UNPACKED: BenchmarkTest = (size: number) => {
  const matrix = test_util.randomArrayInRange(size * size, -1, 1);
  const channelsPerTexture = webgl_util.getChannelsPerTexture();
  const unpackedArray =
      new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(
          matrix.length, channelsPerTexture));
  const start = performance.now();
  for (let i = 0; i < OPS_PER_RUN; ++i) {
    tex_util.encodeMatrixToUnpackedArray(
        matrix, unpackedArray, channelsPerTexture);
  }
  const end = performance.now();
  return (end - start) / OPS_PER_RUN;
};

export const BENCHMARK_ENCODE_PACKED: BenchmarkTest = (size: number) => {
  const matrix = test_util.randomArrayInRange(size * size, -1, 1);
  const packedRGBA = new Float32Array(
      tex_util.getPackedRGBAArraySizeFromMatrixShape(size, size));
  const start = performance.now();
  for (let i = 0; i < OPS_PER_RUN; ++i) {
    tex_util.encodeMatrixToPackedRGBA(matrix, size, size, packedRGBA);
  }
  const end = performance.now();
  return (end - start) / OPS_PER_RUN;
};

export const BENCHMARK_DECODE_UNPACKED: BenchmarkTest = (size: number) => {
  const matrix = test_util.randomArrayInRange(size * size, -1, 1);
  const channelsPerTexture = webgl_util.getChannelsPerTexture();
  const unpackedArray =
      new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(
          matrix.length, channelsPerTexture));
  tex_util.encodeMatrixToUnpackedArray(
      matrix, unpackedArray, channelsPerTexture);
  const start = performance.now();
  for (let i = 0; i < OPS_PER_RUN; ++i) {
    tex_util.decodeMatrixFromUnpackedArray(
        unpackedArray, matrix, channelsPerTexture);
  }
  const end = performance.now();
  return (end - start) / OPS_PER_RUN;
};

export const BENCHMARK_DECODE_PACKED: BenchmarkTest = (size: number) => {
  const matrix = test_util.randomArrayInRange(size * size, -1, 1);
  const packedRGBA = new Float32Array(
      tex_util.getPackedRGBAArraySizeFromMatrixShape(size, size));
  tex_util.encodeMatrixToPackedRGBA(matrix, size, size, packedRGBA);
  const start = performance.now();
  for (let i = 0; i < OPS_PER_RUN; ++i) {
    tex_util.decodeMatrixFromPackedRGBA(packedRGBA, size, size, matrix);
  }
  const end = performance.now();
  return (end - start) / OPS_PER_RUN;
};
