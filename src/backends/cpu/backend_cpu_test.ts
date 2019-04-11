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

import * as tf from '../../index';
import {describeWithFlags} from '../../jasmine_util';
import {tensor2d} from '../../ops/ops';
import {expectArraysEqual} from '../../test_util';

import {MathBackendCPU} from './backend_cpu';
import {CPU_ENVS} from './backend_cpu_test_registry';

describeWithFlags('backendCPU', CPU_ENVS, () => {
  let backend: MathBackendCPU;
  beforeEach(() => {
    backend = tf.backend() as MathBackendCPU;
  });

  it('register empty string tensor', () => {
    const t = tf.Tensor.make([3], {}, 'string');
    expect(backend.readSync(t.dataId) == null).toBe(true);
  });

  it('register empty string tensor and write', () => {
    const t = tf.Tensor.make([3], {}, 'string');
    backend.write(t.dataId, ['c', 'a', 'b']);
    expectArraysEqual(backend.readSync(t.dataId), ['c', 'a', 'b']);
  });

  it('register string tensor with values', () => {
    const t = tf.Tensor.make([3], {values: ['a', 'b', 'c']}, 'string');
    expectArraysEqual(backend.readSync(t.dataId), ['a', 'b', 'c']);
  });

  it('register string tensor with values and overwrite', () => {
    const t = tf.Tensor.make([3], {values: ['a', 'b', 'c']}, 'string');
    backend.write(t.dataId, ['c', 'a', 'b']);
    expectArraysEqual(backend.readSync(t.dataId), ['c', 'a', 'b']);
  });

  it('register string tensor with values and mismatched shape', () => {
    expect(() => tf.tensor(['a', 'b', 'c'], [4], 'string')).toThrowError();
  });
});

describeWithFlags('depthToSpace', CPU_ENVS, () => {
  it('throws when CPU backend used with data format NCHW', () => {
    const t = tf.tensor4d([1, 2, 3, 4], [1, 4, 1, 1]);
    const blockSize = 2;
    const dataFormat = 'NCHW';

    expect(() => tf.depthToSpace(t, blockSize, dataFormat))
        .toThrowError(
            `Only NHWC dataFormat supported on CPU for depthToSpace. Got ${
                dataFormat}`);
  });
});

describeWithFlags('gatherND CPU', CPU_ENVS, () => {
  it('should throw error when index out of range', () => {
    const indices = tensor2d([0, 2, 99], [3, 1], 'int32');
    const input = tensor2d(
        [100, 101, 102, 777, 778, 779, 10000, 10001, 10002], [3, 3], 'float32');
    expect(() => tf.gatherND(input, indices)).toThrow();
  });
});

describeWithFlags('scatterND CPU', CPU_ENVS, () => {
  it('should throw error when index out of range', () => {
    const indices = tf.tensor2d([0, 4, 99], [3, 1], 'int32');
    const updates = tf.tensor2d(
        [100, 101, 102, 777, 778, 779, 10000, 10001, 10002], [3, 3], 'float32');
    const shape = [5, 3];
    expect(() => tf.scatterND(indices, updates, shape)).toThrow();
  });

  it('should throw error when indices has wrong dimension', () => {
    const indices = tf.tensor2d([0, 4, 99], [3, 1], 'int32');
    const updates = tf.tensor2d(
        [100, 101, 102, 777, 778, 779, 10000, 10001, 10002], [3, 3], 'float32');
    const shape = [2, 3];
    expect(() => tf.scatterND(indices, updates, shape)).toThrow();
  });
});

describeWithFlags('sparseToDense CPU', CPU_ENVS, () => {
  it('should throw error when index out of range', () => {
    const defaultValue = 2;
    const indices = tf.tensor1d([0, 2, 6], 'int32');
    const values = tf.tensor1d([100, 101, 102], 'int32');
    const shape = [6];
    expect(() => tf.sparseToDense(indices, values, shape, defaultValue))
        .toThrow();
  });
});

describeWithFlags('memory cpu', CPU_ENVS, () => {
  it('unreliable is true due to auto gc', () => {
    tf.tensor(1);
    const mem = tf.memory();
    expect(mem.numTensors).toBe(1);
    expect(mem.numDataBuffers).toBe(1);
    expect(mem.numBytes).toBe(4);
    expect(mem.unreliable).toBe(true);

    const expectedReason =
        'The reported memory is an upper bound. Due to automatic garbage ' +
        'collection, the true allocated memory may be less.';
    expect(mem.reasons.indexOf(expectedReason) >= 0).toBe(true);
  });

  it('unreliable is true due to both auto gc and string tensors', () => {
    tf.tensor(1);
    tf.tensor('a');

    const mem = tf.memory();
    expect(mem.numTensors).toBe(2);
    expect(mem.numDataBuffers).toBe(2);
    expect(mem.numBytes).toBe(6);
    expect(mem.unreliable).toBe(true);

    const expectedReasonGC =
        'The reported memory is an upper bound. Due to automatic garbage ' +
        'collection, the true allocated memory may be less.';
    expect(mem.reasons.indexOf(expectedReasonGC) >= 0).toBe(true);
    const expectedReasonString =
        'Memory usage by string tensors is approximate ' +
        '(2 bytes per character)';
    expect(mem.reasons.indexOf(expectedReasonString) >= 0).toBe(true);
  });
});
