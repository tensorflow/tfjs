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

import {Array3D, conv_util, NDArrayMathCPU} from '../deeplearn';

import {BenchmarkTest} from './benchmark';

const OP_RUNS = 40;

export const MAX_POOL_BENCHMARK_TEST: BenchmarkTest = (size: number) => {
  if (size > 512) {
    return -1;
  }
  const positions = false;
  return testMaxPool(size, positions);
};

function testMaxPool(size: number, positions: boolean): number {
  const math = new NDArrayMathCPU();
  const outputDepth = 1;
  const xShape: [number, number, number] = [size, size, outputDepth];
  const fieldSize = 11;
  const stride = 1;
  const zeroPad = conv_util.computeDefaultPad(xShape, fieldSize, stride);

  const x = Array3D.randUniform(xShape, -1, 1);

  const start = performance.now();
  for (let i = 0; i < OP_RUNS; i++) {
    math.maxPool(x as Array3D, fieldSize, stride, zeroPad);
  }
  const avgTime = (performance.now() - start) / OP_RUNS;

  x.dispose();

  return avgTime;
}
