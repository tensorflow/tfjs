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

import * as tf from '@tensorflow/tfjs';
// tslint:disable-next-line: no-imports-from-dist
import {TestKernelBackend} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {createTensorsTypeOpAttr, createTypeOpAttr, ensureTensorflowBackend, getTFDType, nodeBackend, NodeJSKernelBackend} from './nodejs_kernel_backend';

describe('delayed upload', () => {
  it('should handle data before op execution', async () => {
    const t = tf.tensor1d([1, 2, 3]);
    tf.test_util.expectArraysClose(await t.data(), [1, 2, 3]);

    const r = t.add(tf.tensor1d([4, 5, 6]));
    tf.test_util.expectArraysClose(await r.data(), [5, 7, 9]);
  });

  it('Should not cache tensors in the tensor map for device support. ', () => {
    const logits = tf.tensor1d([1, 2, 3]);
    const softmaxLogits = tf.softmax(logits);
    const data = softmaxLogits.dataSync();
    expect(softmaxLogits.dataSync()[0]).toEqual(data[0]);
    expect(softmaxLogits.dataSync()[1]).toEqual(data[1]);
    expect(softmaxLogits.dataSync()[2]).toEqual(data[2]);
  });
});

describe('type casting', () => {
  it('exp support int32', () => {
    tf.exp(tf.scalar(2, 'int32'));
  });
});

describe('conv3d dilations', () => {
  it('CPU should throw error on dilations >1', () => {
    const input: tf.Tensor5D = tf.ones([1, 2, 2, 2, 1]);
    const filter: tf.Tensor5D = tf.ones([1, 1, 1, 1, 1]);
    expect(() => {
      tf.conv3d(input, filter, 1, 'same', 'NDHWC', [2, 2, 2]);
    }).toThrowError();
  });
  it('GPU should handle dilations >1', () => {
    // This test can only run locally with CUDA bindings and GPU package
    // installed.
    if ((tf.backend() as NodeJSKernelBackend).isUsingGpuDevice) {
      const input: tf.Tensor5D = tf.ones([1, 2, 2, 2, 1]);
      const filter: tf.Tensor5D = tf.ones([1, 1, 1, 1, 1]);
      tf.conv3d(input, filter, 1, 'same', 'NDHWC', [2, 2, 2]);
    }
  });
});

describe('Exposes Backend for internal Op execution.', () => {
  it('Provides the Node backend over a function', () => {
    const backend = nodeBackend();
    expect(backend instanceof NodeJSKernelBackend).toBeTruthy();
  });

  it('Provides internal access to the binding', () => {
    expect(nodeBackend().binding).toBeDefined();
  });

  it('throw error if backend is not tensorflow', async done => {
    try {
      const testBackend = new TestKernelBackend();
      tf.registerBackend('fake', () => testBackend);
      tf.setBackend('fake');

      ensureTensorflowBackend();
      done.fail();
    } catch (err) {
      expect(err.message)
          .toBe(
              'Expect the current backend to be "tensorflow", but got "fake"');
      tf.setBackend('tensorflow');
      done();
    }
  });
});

describe('getTFDType()', () => {
  const binding = nodeBackend().binding;

  it('handles float32', () => {
    expect(getTFDType('float32')).toBe(binding.TF_FLOAT);
  });
  it('handles int32', () => {
    expect(getTFDType('int32')).toBe(binding.TF_INT32);
  });
  it('handles bool', () => {
    expect(getTFDType('bool')).toBe(binding.TF_BOOL);
  });
  it('handles unknown types', () => {
    expect(() => getTFDType(null)).toThrowError();
  });
});

describe('createTypeOpAttr()', () => {
  const binding = nodeBackend().binding;

  it('Creates a valid type attribute', () => {
    const attr = createTypeOpAttr('foo', 'float32');
    expect(attr.name).toBe('foo');
    expect(attr.type).toBe(binding.TF_ATTR_TYPE);
    expect(attr.value).toBe(binding.TF_FLOAT);
  });

  it('handles unknown dtypes', () => {
    expect(() => createTypeOpAttr('foo', null)).toThrowError();
  });
});

describe('Returns TFEOpAttr for a Tensor or list of Tensors', () => {
  const binding = nodeBackend().binding;

  it('handles a single Tensor', () => {
    const result = createTensorsTypeOpAttr('T', tf.scalar(13, 'float32'));
    expect(result.name).toBe('T');
    expect(result.type).toBe(binding.TF_ATTR_TYPE);
    expect(result.value).toBe(binding.TF_FLOAT);
  });
  it('handles a list of Tensors', () => {
    const tensors = [tf.scalar(1, 'int32'), tf.scalar(20.1, 'float32')];
    const result = createTensorsTypeOpAttr('T', tensors);
    expect(result.name).toBe('T');
    expect(result.type).toBe(binding.TF_ATTR_TYPE);
    expect(result.value).toBe(binding.TF_INT32);
  });
  it('handles null', () => {
    expect(() => createTensorsTypeOpAttr('T', null)).toThrowError();
  });
  it('handles list of null', () => {
    const inputs = [null, null] as tf.Tensor[];
    expect(() => createTensorsTypeOpAttr('T', inputs)).toThrowError();
  });
});
