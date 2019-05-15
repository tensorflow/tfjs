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

// Maximum number of time before CPU tests don't execute during the next round.
export const LAST_RUN_CPU_CUTOFF_MS = 5000;

export async function benchmark(benchmarkFn: () => tf.Tensor | tf.Tensor[]):
    Promise<number> {
  // Use normal performance.now() timing even though query timers are enabled
  // again because we want to account for more than just GPU time.
  const start = performance.now();
  const result = benchmarkFn();
  await result.data();
  return performance.now() - start;
}

export async function asyncBenchmark(
    asyncBenchmarkFn: () =>
        Promise<tf.Tensor>| Promise<tf.Tensor[]>): Promise<number> {
  const start = performance.now();
  const result = await asyncBenchmarkFn() as tf.Tensor[] | tf.Tensor;

  const outRes = Array.isArray(result) ? (result as tf.Tensor[])[0] : result;
  await (outRes as tf.Tensor).data();
  (outRes as tf.Tensor).dispose();

  return performance.now() - start;
}
