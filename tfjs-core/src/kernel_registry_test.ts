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
import {KernelBackend} from './index';
import {ALL_ENVS, describeWithFlags} from './jasmine_util';
import {KernelFunc, TensorInfo} from './kernel_registry';
import {expectArraysClose} from './test_util';

describeWithFlags('kernel_registry', ALL_ENVS, () => {
  it('register a kernel and call it', () => {
    let called = false;
    tf.registerKernel({
      kernelName: 'MyKernel',
      backendName: tf.getBackend(),
      kernelFunc: ({inputs, attrs}) => {
        expect(attrs.a).toBe(5);
        expect(inputs.x.shape).toEqual([2, 2]);
        expect(inputs.x.dtype).toBe('float32');
        called = true;
        return {dtype: 'float32', shape: [3, 3], dataId: {}};
      }
    });

    const inputs = {x: tf.zeros([2, 2])};
    const attrs = {a: 5};
    const res = tf.engine().runKernel('MyKernel', inputs, attrs) as TensorInfo;

    expect(called).toBe(true);
    expect(res.dtype).toBe('float32');
    expect(res.shape).toEqual([3, 3]);

    tf.unregisterKernel('MyKernel', tf.getBackend());
  });

  it('errors when running non-existent kernel', () => {
    const inputs = {};
    const attrs = {};
    expect(() => tf.engine().runKernel('DoesNotExist', inputs, attrs))
        .toThrowError();
  });

  it('errors when registering the same kernel twice', () => {
    tf.registerKernel({
      kernelName: 'MyKernel',
      backendName: tf.getBackend(),
      kernelFunc: () => {
        return null;
      }
    });
    expect(() => tf.registerKernel({
      kernelName: 'MyKernel',
      backendName: tf.getBackend(),
      kernelFunc: () => {
        return null;
      }
    })).toThrowError();

    tf.unregisterKernel('MyKernel', tf.getBackend());
  });

  it('register same kernel on two different backends', () => {
    interface TestBackend extends KernelBackend {
      id: number;
    }
    tf.registerBackend('backend1', () => {
      return {
        id: 1,
        dispose: () => null,
        disposeData: (dataId: {}) => null,
        numDataIds: () => 0
      } as TestBackend;
    });
    tf.registerBackend('backend2', () => {
      return {
        id: 2,
        dispose: () => null,
        disposeData: (dataId: {}) => null,
        numDataIds: () => 0
      } as TestBackend;
    });

    let lastStorageId = -1;
    const kernelFunc: KernelFunc = ({backend}) => {
      lastStorageId = (backend as TestBackend).id;
      return {dataId: {}, shape: [], dtype: 'float32'};
    };
    tf.registerKernel(
        {kernelName: 'MyKernel', backendName: 'backend1', kernelFunc});
    tf.registerKernel(
        {kernelName: 'MyKernel', backendName: 'backend2', kernelFunc});

    // No kernel has been executed yet.
    expect(lastStorageId).toBe(-1);

    // Kernel was executed on the first backend.
    tf.setBackend('backend1');
    tf.engine().runKernel('MyKernel', {}, {});
    expect(lastStorageId).toBe(1);

    // Kernel was executed on the second backend.
    tf.setBackend('backend2');
    tf.engine().runKernel('MyKernel', {}, {});
    expect(lastStorageId).toBe(2);

    tf.removeBackend('backend1');
    tf.removeBackend('backend2');
    tf.unregisterKernel('MyKernel', 'backend1');
    tf.unregisterKernel('MyKernel', 'backend2');
  });

  it('register kernel with setup and dispose functions', () => {
    const backendName = 'custom-backend';
    const kernelName = 'MyKernel';
    interface TestBackend extends KernelBackend {}
    const customBackend = {
      dispose: () => null,
      disposeData: (dataId: {}) => null,
      numDataIds: () => 0
    } as TestBackend;
    tf.registerBackend(backendName, () => customBackend);

    const kernelFunc: KernelFunc = () => {
      return {dataId: {}, shape: [], dtype: 'float32'};
    };
    let setupCalled = false;
    const setupFunc = (backend: KernelBackend) => {
      expect(backend).toBe(customBackend);
      setupCalled = true;
    };
    let disposeCalled = false;
    const disposeFunc = (backend: KernelBackend) => {
      expect(backend).toBe(customBackend);
      disposeCalled = true;
    };
    tf.registerKernel(
        {kernelName, backendName, kernelFunc, setupFunc, disposeFunc});

    expect(setupCalled).toBe(false);
    expect(disposeCalled).toBe(false);

    tf.setBackend(backendName);
    expect(setupCalled).toBe(true);
    expect(disposeCalled).toBe(false);

    // Kernel was executed on the first backend.
    tf.engine().runKernel(kernelName, {}, {});

    tf.removeBackend(backendName);
    expect(setupCalled).toBe(true);
    expect(disposeCalled).toBe(true);

    tf.unregisterKernel(kernelName, backendName);
  });
});

describeWithFlags('gradient registry', ALL_ENVS, () => {
  it('register a kernel with gradient and call it', async () => {
    let kernelWasCalled = false;
    let gradientWasCalled = false;
    const kernelName = 'MyKernel';
    const x = tf.zeros([2, 2]);

    tf.registerKernel({
      kernelName,
      backendName: tf.getBackend(),
      kernelFunc: () => {
        kernelWasCalled = true;
        return {dtype: 'float32', shape: [3, 3], dataId: {}};
      }
    });

    tf.registerGradient({
      kernelName,
      gradFunc: (dy: tf.Tensor, saved) => {
        // Make sure saved input (x) was passed to the gradient function.
        expect(saved[0].dataId).toEqual(x.dataId);
        // Make sure dy matches the shape of the output.
        expect(dy.shape).toEqual([3, 3]);
        gradientWasCalled = true;
        return {x: () => tf.fill([2, 2], 3)};
      },
    });

    const gradFunc = tf.grad(
        x => tf.engine().runKernel(
                 kernelName, {x}, {} /* attrs */, [x] /* inputsToSave */) as
            tf.Tensor);
    const dx = gradFunc(x);
    expect(kernelWasCalled).toBe(true);
    expect(gradientWasCalled).toBe(true);
    expect(dx.dtype).toBe('float32');
    expect(dx.shape).toEqual([2, 2]);
    expectArraysClose(await dx.data(), [3, 3, 3, 3]);
    tf.unregisterKernel(kernelName, tf.getBackend());
    tf.unregisterGradient(kernelName);
  });

  it('errors when running non-existent gradient', () => {
    const kernelName = 'MyKernel';
    const x = tf.zeros([2, 2]);

    tf.registerKernel({
      kernelName,
      backendName: tf.getBackend(),
      kernelFunc: () => ({dtype: 'float32', shape: [3, 3], dataId: {}})
    });

    const gradFunc = tf.grad(
        x => tf.engine().runKernel(
                 kernelName, {x}, {} /* attrs */, [x] /* inputsToSave */) as
            tf.Tensor);
    expect(() => gradFunc(x))
        .toThrowError(/gradient function not found for MyKernel/);

    tf.unregisterKernel(kernelName, tf.getBackend());
  });

  it('warning when registering the same gradient twice', () => {
    const kernelName = 'MyKernel';
    tf.registerGradient({kernelName, gradFunc: () => null});
    spyOn(console, 'warn').and.callFake((msg: string) => {
      expect(msg).toBe('Overriding the gradient for \'MyKernel\'');
    });
    tf.registerGradient({kernelName, gradFunc: () => null});
    tf.unregisterGradient(kernelName);
  });
});
