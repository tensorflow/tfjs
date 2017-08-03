/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {NDArrayMathCPU} from '../../src/math/math_cpu';
import {Array2D, NDArray} from '../../src/math/ndarray';

import {BenchmarkTest} from './benchmark';

const OPS_PER_SMALL_RUN = 1;

export const BENCHMARK_TEST: BenchmarkTest = (size: number) => {
  if (size > 512) {
    return -1;
  }
  const math = new NDArrayMathCPU();
  const a = NDArray.randUniform<Array2D>([size, size], -1, 1);
  const b = NDArray.randUniform<Array2D>([size, size], -1, 1);
  const runs = (size < 192) ? OPS_PER_SMALL_RUN : 1;
  const start = performance.now();
  for (let i = 0; i < runs; i++) {
    math.matMul(a, b);
  }
  const end = performance.now();
  return (end - start) / runs;
};
