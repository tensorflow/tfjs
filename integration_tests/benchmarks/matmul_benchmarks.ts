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
import * as tf from '@tensorflow/tfjs-core';

import {BenchmarkTest} from './types';
import * as util from './util';

export class MatmulCPUBenchmark implements BenchmarkTest {
  lastRunTimeMs: number;
  async run(size: number): Promise<number> {
    if (this.lastRunTimeMs > util.LAST_RUN_CPU_CUTOFF_MS) {
      return new Promise<number>((resolve, reject) => {
        resolve(-1);
      });
    }
    tf.setBackend('cpu');

    const a: tf.Tensor2D = tf.randomUniform([size, size], -1, 1);
    const b: tf.Tensor2D = tf.randomUniform([size, size], -1, 1);
    const start = performance.now();
    tf.matMul(a, b);
    const end = performance.now();
    this.lastRunTimeMs = end - start;
    return this.lastRunTimeMs;
  }
}

export class MatmulGPUBenchmark implements BenchmarkTest {
  async run(size: number): Promise<number> {
    tf.setBackend('webgl');

    const a: tf.Tensor2D = tf.randomNormal([size, size]);
    const b: tf.Tensor2D = tf.randomNormal([size, size]);

    const benchmark = () => tf.matMul(a, b);

    const time = await util.benchmark(benchmark);

    a.dispose();
    b.dispose();

    return time;
  }
}
