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

import {CoCoSSDBenchmark} from './cocossd_benchmarks';
import {ConvGPUBenchmark, RegularConvParams} from './conv_benchmarks';
import {MatmulGPUBenchmark} from './matmul_benchmarks';
import {MobileNetV1GPUBenchmark} from './mobilenet_benchmarks';
import * as test_util from './test_util';
import {UniversalSentenceEncoderBenchmark} from './use_benchmarks';

const BENCHMARK_RUNS = 100;

describe('benchmarks', () => {
  console.log('INSIDE BENCHMARKS YAY');
  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 600000;
  });

  it('matmul', async () => {
    console.log('INSIDE MATMUL');
    const sizes = [1, 100, 400, 1000];

    const benchmark = new MatmulGPUBenchmark();

    await test_util.benchmarkAndLog(
        'matmul', size => benchmark.run(size), sizes, size => `N=${size}`,
        BENCHMARK_RUNS);
  });

  it('conv2d', async () => {
    console.log('INSIDE CONV2D');
    const sizes = [10, 100, 200];
    const convParams: RegularConvParams =
        {inDepth: 16, outDepth: 32, filterSize: 5, stride: 1, pad: 'same'};
    const benchmark = new ConvGPUBenchmark();

    await test_util.benchmarkAndLog(
        'conv2d', size => benchmark.run(size, 'regular', convParams), sizes,
        size => `N=${size} ${JSON.stringify(convParams)}`, BENCHMARK_RUNS);
  });

  it('mobilenet_v1', async () => {
    console.log('INSIDE MOBILENET');
    const sizes = [1];  // MobileNet version

    const benchmark = new MobileNetV1GPUBenchmark();

    await test_util.benchmarkAndLog(
        'mobilenet_v1', size => benchmark.run(size), sizes,
        size => `N=${size}_0_224`, BENCHMARK_RUNS);
  });

  it('use', async () => {
    const sizes = [1];
    const runs = 20;

    const benchmark = new UniversalSentenceEncoderBenchmark();

    await test_util.benchmarkAndLog(
        'use', size => benchmark.run(size), sizes, size => '41', runs);
  });

  it('cocossd', async () => {
    const sizes = [1];
    const runs = 10;

    const benchmark = new CoCoSSDBenchmark();

    await test_util.benchmarkAndLog(
        'cocossd', size => benchmark.run(size), sizes, size => '224', runs);
  });
});
