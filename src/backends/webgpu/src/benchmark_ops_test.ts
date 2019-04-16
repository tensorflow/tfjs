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

import * as tf from './index';

xdescribe('Ops benchmarks', () => {
  it('matMul', async () => {
    await tf.ready;

    const times = [];

    const a = tf.randomNormal([500, 500]);
    const b = tf.randomNormal([500, 500]);

    let c = tf.matMul(a, b);
    await c.data();

    for (let i = 0; i < 100; i++) {
      const start = performance.now();
      c = tf.matMul(a, b);
      await c.data();
      times.push(performance.now() - start);
    }

    console.log(
        `Average time ms: ${times.reduce((a, b) => a + b, 0) / times.length}`);
    console.log(`Min time ms: ${Math.min(...times)}`);
  });
});