/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {ALL_ENVS, describeWithFlags} from './jasmine_util';

function isFloat(num: number) {
  return num % 1 !== 0;
}

describeWithFlags('nextFrame', ALL_ENVS, () => {
  it('basic usage', async () => {
    const t0 = tf.util.now();
    await tf.nextFrame();
    const t1 = tf.util.now();

    // tf.nextFrame may take no more than 1ms to complete, so this test is
    // meaningful only if the precision of tf.util.now is better than 1ms.
    // After version 59, the precision of Firefox's tf.util.now becomes 2ms by
    // default for security issues, https://caniuse.com/?search=performance.now.
    // Then, this test is dropped for Firefox, even though it could be
    // set to better precision through browser setting,
    // https://github.com/lumen/threading-benchmarks/issues/7.
    if (isFloat(t0) || isFloat(t1)) {
      // If t0 or t1 have decimal point, it means the precision is better than
      // 1ms.
      expect(t1).toBeGreaterThan(t0);
    }
  });

  it('does not block timers', async () => {
    let flag = false;
    setTimeout(() => {
      flag = true;
    }, 50);
    const t0 = tf.util.now();
    expect(flag).toBe(false);
    while (tf.util.now() - t0 < 1000 && !flag) {
      await tf.nextFrame();
    }
    expect(flag).toBe(true);
  });
});
