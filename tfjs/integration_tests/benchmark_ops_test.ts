/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {ConvGPUBenchmark, RegularConvParams} from './conv_benchmarks';
import {MatmulGPUBenchmark} from './matmul_benchmarks';
import * as test_util from './test_util';

const BENCHMARK_RUNS = 100;

describe('benchmark ops', () => {
  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });

  it('matmul', async () => {
    const sizes = [1, 100, 400, 1000];

    const benchmark = new MatmulGPUBenchmark();

    await test_util.benchmarkAndLog(
        'matmul', size => benchmark.run(size), sizes, size => `N=${size}`,
        BENCHMARK_RUNS);
  });

  it('conv2d', async () => {
    const sizes = [10, 100, 200];
    const convParams: RegularConvParams =
        {inDepth: 16, outDepth: 32, filterSize: 5, stride: 1, pad: 'same'};
    const benchmark = new ConvGPUBenchmark();

    await test_util.benchmarkAndLog(
        'conv2d', size => benchmark.run(size, 'regular', convParams), sizes,
        size => `N=${size} ${JSON.stringify(convParams)}`, BENCHMARK_RUNS);
  });
});
