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

import {BenchmarkRun, BenchmarkRunGroup} from './benchmark';
import * as conv_gpu_benchmark from './conv_gpu_benchmark';
import * as conv_transpose_gpu_benchmark from './conv_transpose_gpu_benchmark';
import * as logsumexp_cpu_benchmark from './logsumexp_cpu_benchmark';
import * as logsumexp_gpu_benchmark from './logsumexp_gpu_benchmark';
import * as max_pool_backprop_gpu_benchmark from './max_pool_backprop_gpu_benchmark';
import * as max_pool_gpu_benchmark from './max_pool_gpu_benchmark';
import * as mulmat_cpu_benchmark from './mulmat_cpu_benchmark';
import * as mulmat_gpu_benchmark from './mulmat_gpu_benchmark';
import * as tex_util_benchmark from './tex_util_benchmark';

export const BENCHMARK_RUN_GROUPS: BenchmarkRunGroup[] = [
  {
    name:
        'Matrix Multiplication (CPU vs GPU): matmul([size, size], [size, size])',
    min: 0,
    max: 1024,
    stepSize: 64,
    stepToSizeTransformation: (step: number) => Math.max(1, step),
    benchmarkRuns: [
      new BenchmarkRun('mulmat_gpu', mulmat_gpu_benchmark.BENCHMARK_TEST),
      new BenchmarkRun('mulmat_cpu', mulmat_cpu_benchmark.BENCHMARK_TEST)
    ],
  },
  {
    name: 'Convolution (GPU): conv over image [size, size, 1]',
    min: 0,
    max: 1024,
    stepSize: 64,
    stepToSizeTransformation: (step: number) => Math.max(1, step),
    benchmarkRuns: [new BenchmarkRun(
        'd1=1, d2=1, f=11, s=1', conv_gpu_benchmark.BENCHMARK_TEST)],
  },
  {
    name: 'Convolution Transposed (GPU): deconv over image [size, size, 1]',
    min: 0,
    max: 1024,
    stepSize: 64,
    stepToSizeTransformation: (step: number) => Math.max(1, step),
    benchmarkRuns: [new BenchmarkRun(
        'd1=1, d2=1, f=11, s=1', conv_transpose_gpu_benchmark.BENCHMARK_TEST)],
  },
  {
    name: 'Max pool (GPU)',
    min: 0,
    max: 1024,
    stepSize: 64,
    stepToSizeTransformation: (step: number) => Math.max(1, step),
    benchmarkRuns: [new BenchmarkRun(
        'd1=1, d2=1, f=11, s=1',
        max_pool_gpu_benchmark.MAX_POOL_BENCHMARK_TEST)],
  },
  {
    name: 'LogSumExp (CPU vs GPU): input [size, size]',
    min: 0,
    max: 1024,
    stepSize: 64,
    stepToSizeTransformation: (step: number) => Math.max(1, step),
    benchmarkRuns: [
      new BenchmarkRun('logsumexp_gpu', logsumexp_gpu_benchmark.BENCHMARK_TEST),
      new BenchmarkRun('logsumexp_cpu', logsumexp_cpu_benchmark.BENCHMARK_TEST)
    ],
  }
];
