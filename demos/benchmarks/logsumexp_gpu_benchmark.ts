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

import {GPGPUContext} from '../../src/math/webgl/gpgpu_context';
import * as logsumexp_gpu from '../../src/math/webgl/logsumexp_gpu';
import * as test_util from '../../src/test_util';

import {BenchmarkTest} from './benchmark';

const OP_RUNS = 100;

export const BENCHMARK_TEST: BenchmarkTest = (size: number) => {
  const gpgpu = new GPGPUContext();

  const program =
      gpgpu.createProgram(logsumexp_gpu.getFragmentShaderSource(size, size));

  const aTexture = gpgpu.createMatrixTexture(size, size);
  const resultTexture = gpgpu.createMatrixTexture(size, size);

  const a = test_util.randomArrayInRange(size * size, -1, 1);
  gpgpu.uploadMatrixToTexture(aTexture, size, size, a);

  const start = performance.now();
  for (let i = 0; i < OP_RUNS; i++) {
    logsumexp_gpu.logSumExp(
        gpgpu, program, aTexture, size, size, resultTexture);
  }

  gpgpu.downloadMatrixFromTexture(resultTexture, size, size);
  const avgTime = (performance.now() - start) / OP_RUNS;

  gpgpu.deleteMatrixTexture(aTexture);
  gpgpu.deleteMatrixTexture(resultTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();

  return avgTime;
};
