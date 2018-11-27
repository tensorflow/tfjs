/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import {expectArraysEqual} from '../test_util';

import {KernelBackend} from './backend';
import {MathBackendCPU} from './backend_cpu';

describe('backendCPU', () => {
  let prevBackend: string;
  let backend: KernelBackend;

  beforeAll(() => {
    prevBackend = tf.getBackend();
  });

  beforeEach(() => {
    backend = new MathBackendCPU();
    tf.ENV.registerBackend('test-storage', () => backend);
    tf.setBackend('test-storage');
  });

  afterEach(() => {
    backend.dispose();
    tf.setBackend(prevBackend);
    tf.ENV.removeBackend('test-storage');
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
    expect(() => tf.Tensor.make([4], {values: ['a', 'b', 'c']}, 'string'))
        .toThrowError();
  });
});
