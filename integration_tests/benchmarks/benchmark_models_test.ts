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

// import {CoCoSSDBenchmark} from './cocossd_benchmarks';
import {MobileNetV1GPUBenchmark} from './mobilenet_benchmarks';
import * as test_util from './test_util';
import {UniversalSentenceEncoderBenchmark} from './use_benchmarks';

describe('benchmark models', () => {
  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });

  it('mobilenet_v1', async () => {
    const sizes = [1];  // MobileNet version
    const runs = 20;

    const benchmark = new MobileNetV1GPUBenchmark();
    await benchmark.loadModel();

    await test_util.benchmarkAndLog(
        'mobilenet_v1', size => benchmark.run(size), sizes,
        size => `N=${size}_0_224`, runs);
  });

  it('use', async () => {
    const sizes = [1];
    const runs = 20;

    const benchmark = new UniversalSentenceEncoderBenchmark();
    await benchmark.loadModel();

    await test_util.benchmarkAndLog(
        'use', size => benchmark.run(size), sizes, size => '41', runs);
  });

  // TODO(annyuan): Figure out the invalid texture shape error and
  // enable the benchmark.
  // it('cocossd', async () => {
  //   const sizes = [1];
  //   const runs = 20;

  //   const benchmark = new CoCoSSDBenchmark();
  //   await benchmark.loadModel();

  //   await test_util.benchmarkAndLog(
  //       'cocossd', size => benchmark.run(size), sizes, size => '224', runs);
  // });
});
