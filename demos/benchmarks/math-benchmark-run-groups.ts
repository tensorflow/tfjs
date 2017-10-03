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

import {BenchmarkRun, BenchmarkRunGroup} from './benchmark';
import {ConvBenchmarkParams, ConvGPUBenchmark} from './conv_benchmarks';
// tslint:disable-next-line:max-line-length
import {ConvTransposedBenchmarkParams, ConvTransposedGPUBenchmark} from './conv_transposed_benchmarks';
// tslint:disable-next-line:max-line-length
import {LogSumExpCPUBenchmark, LogSumExpGPUBenchmark} from './logsumexp_benchmarks';
import {MatmulCPUBenchmark, MatmulGPUBenchmark} from './matmul_benchmarks';
// tslint:disable-next-line:max-line-length
import {PoolBenchmarkParams, PoolCPUBenchmark, PoolGPUBenchmark} from './pool_benchmarks';
// tslint:disable-next-line:max-line-length
import {ReductionOpsCPUBenchmark, ReductionOpsGPUBenchmark} from './reduction_ops_benchmark';
import {UnaryOpsCPUBenchmark, UnaryOpsGPUBenchmark} from './unary_ops_benchmark';

export function getRunGroups(): BenchmarkRunGroup[] {
  const groups: BenchmarkRunGroup[] = [];

  groups.push({
    name: 'Matrix Multiplication: ' +
        'matmul([size, size], [size, size])',
    min: 0,
    max: 1024,
    stepSize: 64,
    stepToSizeTransformation: (step: number) => Math.max(1, step),
    benchmarkRuns: [
      new BenchmarkRun('mulmat_gpu', new MatmulGPUBenchmark()),
      new BenchmarkRun('mulmat_cpu', new MatmulCPUBenchmark())
    ],
    params: {}
  });

  const convParams:
      ConvBenchmarkParams = {inDepth: 8, outDepth: 3, filterSize: 7, stride: 1};
  groups.push({
    name: 'Convolution: image [size, size]',
    min: 0,
    max: 1024,
    stepSize: 64,
    stepToSizeTransformation: (step: number) => Math.max(1, step),
    benchmarkRuns:
        [new BenchmarkRun('conv_gpu', new ConvGPUBenchmark(convParams))],
    params: convParams
  });

  const convTransposedParams: ConvTransposedBenchmarkParams =
      {inDepth: 8, outDepth: 3, filterSize: 7, stride: 1};
  groups.push({
    name: 'Convolution Transposed: deconv over image [size, size]',
    min: 0,
    max: 1024,
    stepSize: 64,
    stepToSizeTransformation: (step: number) => Math.max(1, step),
    benchmarkRuns: [new BenchmarkRun(
        'conv_transpose_gpu',
        new ConvTransposedGPUBenchmark(convTransposedParams))],
    params: convTransposedParams
  });

  const maxPoolParams:
      PoolBenchmarkParams = {depth: 8, fieldSize: 4, stride: 4, type: 'max'};
  groups.push({
    name: 'Max pool',
    min: 0,
    max: 1024,
    stepSize: 64,
    stepToSizeTransformation: (step: number) => Math.max(4, step),
    benchmarkRuns: [
      new BenchmarkRun('max_pool_gpu', new PoolGPUBenchmark(maxPoolParams)),
      new BenchmarkRun('max_pool_cpu', new PoolCPUBenchmark(maxPoolParams))
    ],
    params: maxPoolParams
  });

  const avgPoolParams:
      PoolBenchmarkParams = {depth: 8, fieldSize: 4, stride: 4, type: 'avg'};
  groups.push({
    name: 'Avg pool',
    min: 0,
    max: 1024,
    stepSize: 64,
    stepToSizeTransformation: (step: number) => Math.max(4, step),
    benchmarkRuns: [
      new BenchmarkRun('avg_pool_gpu', new PoolGPUBenchmark(avgPoolParams)),
      new BenchmarkRun('avg_pool_cpu', new PoolCPUBenchmark(avgPoolParams))
    ],
    params: avgPoolParams
  });

  groups.push({
    name: 'LogSumExp: input [size, size]',
    min: 0,
    max: 1024,
    stepSize: 64,
    stepToSizeTransformation: (step: number) => Math.max(1, step),
    benchmarkRuns: [
      new BenchmarkRun('logsumexp_gpu', new LogSumExpGPUBenchmark()),
      new BenchmarkRun('logsumexp_cpu', new LogSumExpCPUBenchmark())
    ],
    params: {}
  });

  groups.push({
    name: 'Unary Op Benchmark (CPU vs GPU): input [size, size]',
    min: 0,
    max: 1024,
    stepToSizeTransformation: (step: number) => Math.max(1, step),
    options: [
      'log', 'exp', 'neg', 'sqrt', 'abs', 'relu', 'sigmoid', 'sin', 'cos',
      'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'logSumExp'
    ],
    selectedOption: 'log',
    stepSize: 64,
    benchmarkRuns: [
      new BenchmarkRun('unary ops CPU', new UnaryOpsCPUBenchmark()),
      new BenchmarkRun('unary ops GPU', new UnaryOpsGPUBenchmark())
    ],
    params: {}
  });

  groups.push({
    name: 'Reduction Op Benchmark (CPU vs GPU): input [size, size]',
    min: 0,
    max: 1024,
    stepToSizeTransformation: (step: number) => Math.max(1, step),
    options: ['max', 'min', 'sum'],
    selectedOption: 'max',
    stepSize: 64,
    benchmarkRuns: [
      new BenchmarkRun('reduction ops CPU', new ReductionOpsCPUBenchmark()),
      new BenchmarkRun('reduction ops GPU', new ReductionOpsGPUBenchmark())
    ],
    params: {}
  });

  return groups;
}
