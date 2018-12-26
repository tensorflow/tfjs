/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

describe('Ops are exported from index and work', () => {
  it('tf.mul works', () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor1d([3, 4, 5]);
    tf.test_util.expectArraysClose(a.mul(b), [3, 8, 15]);
  });
});

describe('packages merge', () => {
  it('versions', () => {
    expect(tf.version['tfjs']).toBeDefined();
    expect(tf.version['tfjs-core']).toBeDefined();

    const expectedNodeVersion =
        // tslint:disable-next-line:no-require-imports
        require('../package.json').version;
    expect(tf.version['tfjs-node']).toBe(expectedNodeVersion);
  });

  it('symbols merge', () => {
    expect(tf.mul).toBeDefined();
    expect(tf.layers.dense).toBeDefined();
    expect(tf.io.listModels).toBeDefined();
    expect(tf.io.nodeHTTPRequest).toBeDefined();
  });
});
