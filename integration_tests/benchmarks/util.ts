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

export async function warmupAndBenchmarkGPU(benchmark: () => tf.Tensor):
    Promise<number> {
  // Warmup.
  const out = benchmark();
  await out.data();
  out.dispose();

  const start = performance.now();
  const result = benchmark();
  await result.data();
  return performance.now() - start;
  // return (await tf.time(benchmark)).kernelMs;
}

export async function warmupAndAsyncBenchmarkGPU(
    asyncBenchmark: () =>
        Promise<tf.Tensor[]>| Promise<tf.Tensor>): Promise<number> {
  const out = await asyncBenchmark() as tf.Tensor[] | tf.Tensor;

  if ((out as tf.Tensor[]).length) {
    await ((out as tf.Tensor[])[0] as tf.Tensor).data();
    ((out as tf.Tensor[])[0] as tf.Tensor).dispose();
  } else {
    await (out as tf.Tensor).data();
    (out as tf.Tensor).dispose();
  }

  const start = performance.now();
  const result = await asyncBenchmark() as tf.Tensor[] | tf.Tensor;

  if ((result as tf.Tensor[]).length) {
    await ((result as tf.Tensor[])[0] as tf.Tensor).data();
    ((result as tf.Tensor[])[0] as tf.Tensor).dispose();
  } else {
    await (result as tf.Tensor).data();
    (result as tf.Tensor).dispose();
  }
  return performance.now() - start;
}
