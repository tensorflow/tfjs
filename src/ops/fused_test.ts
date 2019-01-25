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

import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
import {ALL_ENVS, expectArraysClose} from '../test_util';

describeWithFlags('fused matmul', ALL_ENVS, () => {
  it('A x B', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.fused.matMul(a, b);

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(c, [0, 8, -3, 20]);
  });

  it('A x B with relu', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.fused.matMul(a, b, false, false, null, 'relu');

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(c, [0, 8, 0, 20]);
  });

  it('A x B with relu transpose', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [2, 3]);

    const c = tf.fused.matMul(a, b, false, true, null, 'relu');

    expect(c.shape).toEqual([2, 2]);
    expectArraysClose(c, [0, 9, 0, 24]);
  });

  it('A x B with relu and bias', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);
    const c = tf.tensor2d([1, 1, 1, 1], [2, 2]);

    const d = tf.fused.matMul(a, b, false, false, c, 'relu');

    expect(d.shape).toEqual([2, 2]);
    expectArraysClose(d, [1, 9, 0, 21]);
  });

  it('A x B with relu and broadcasted bias', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);
    const c = tf.tensor1d([1, 1]);
    const act: tf.fused.Activation = 'relu';

    const d = tf.fused.matMul(a, b, false, false, c, act);

    expect(d.shape).toEqual([2, 2]);
    expectArraysClose(d, [1, 9, 0, 21]);
  });

  it('A x B with relu and broadcasted bias different rank', () => {
    const a = tf.tensor3d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [2, 2, 3]);
    const b = tf.tensor3d([0, 1, -3, 2, 2, 1, 0, 1, -3, 2, 2, 1], [2, 3, 2]);
    const c = tf.tensor2d([1, 2], [1, 2]);
    const act: tf.fused.Activation = 'relu';

    const d = tf.fused.matMul(a, b, false, false, c, act);

    expect(d.shape).toEqual([2, 2, 2]);
    expectArraysClose(d, [2, 6, 0, 18, 0, 30, 0, 42]);
  });

  it('A x B with bias only', () => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);
    const c = tf.tensor2d([1, 1, 1, 1], [2, 2]);

    const d = tf.fused.matMul(a, b, false, false, c, 'linear');

    expect(d.shape).toEqual([2, 2]);
    expectArraysClose(d, [1, 9, -2, 21]);
  });

  it('A x B with relu gradient', () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, -30], [2, 3]);
    const b = tf.tensor2d([2, 3, 4, -1, 2, 3], [3, 2]);
    const dy = tf.tensor2d([1, 10, 20, 30], [2, 2]);

    const grads = tf.grads((a, b) => {
      const prod = tf.matMul(a, b, false, false);
      return tf.relu(prod);
    });

    const fusedGrads = tf.grads((a, b) => {
      return tf.fused.matMul(a, b, false, false, null, 'relu');
    });

    const [da, db] = grads([a, b], dy);
    const [fusedDa, fusedDb] = fusedGrads([a, b], dy);
    expectArraysClose(da, fusedDa);
    expectArraysClose(db, fusedDb);
  });

  it('A x B with relu bias gradient', () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, -30], [2, 3]);
    const b = tf.tensor2d([2, 3, 4, -1, 2, 3], [3, 2]);
    const c = tf.tensor2d([1, 1, 1, 1], [2, 2]);

    const dy = tf.tensor2d([1, 10, 20, 30], [2, 2]);

    const grads = tf.grads((a, b, c) => {
      const prod = tf.matMul(a, b, false, false);
      const sum = tf.add(prod, c);
      return tf.relu(sum);
    });

    const fusedGrads = tf.grads((a, b, c) => {
      return tf.fused.matMul(a, b, false, false, c, 'relu');
    });

    const [da, db, dc] = grads([a, b, c], dy);
    const [fusedDa, fusedDb, fusedDc] = fusedGrads([a, b, c], dy);

    expectArraysClose(da, fusedDa);
    expectArraysClose(db, fusedDb);
    expectArraysClose(dc, fusedDc);
  });

  it('A x B with relu bias gradient transpose', () => {
    const a = tf.tensor2d([1, 2, 3, 10, 20, -30], [3, 2]);
    const b = tf.tensor2d([2, 3, 4, -1, 2, 3], [3, 2]);
    const c = tf.tensor2d([1, 1, 1, 1], [2, 2]);

    const dy = tf.tensor2d([1, 10, 20, 30], [2, 2]);

    const grads = tf.grads((a, b, c) => {
      const prod = tf.matMul(a, b, true, false);
      const sum = tf.add(prod, c);
      return tf.relu(sum);
    });

    const fusedGrads = tf.grads((a, b, c) => {
      return tf.fused.matMul(a, b, true, false, c, 'relu');
    });

    const [da, db, dc] = grads([a, b, c], dy);
    const [fusedDa, fusedDb, fusedDc] = fusedGrads([a, b, c], dy);

    expectArraysClose(da, fusedDa);
    expectArraysClose(db, fusedDb);
    expectArraysClose(dc, fusedDc);
  });
});