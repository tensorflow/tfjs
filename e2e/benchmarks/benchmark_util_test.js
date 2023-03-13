/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

/**
 * The unit tests in this file can be run by opening `./SpecRunner.html` in
 * browser.
 */

function sleep(timeMs) {
  return new Promise(resolve => setTimeout(resolve, timeMs));
}

describe('benchmark_util', () => {
  beforeEach(() => tf.setBackend('cpu'));

  describe('generateInput', () => {
    it('LayersModel', () => {
      const model = tf.sequential(
          {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
      const input = generateInput(model);
      expect(input.length).toEqual(1);
      expect(input[0]).toBeInstanceOf(tf.Tensor);
      expect(input[0].shape).toEqual([1, 3]);
    });
  });

  describe('generateInputFromDef', () => {
    it('should respect int32 data range', async () => {
      const inputDef =
          [{shape: [1, 1, 10], dtype: 'int32', range: [0, 255], name: 'test'}];
      const input = generateInputFromDef(inputDef);
      expect(input.length).toEqual(1);
      expect(input[0]).toBeInstanceOf(tf.Tensor);
      expect(input[0].shape).toEqual([1, 1, 10]);
      expect(input[0].dtype).toEqual('int32');
      const data = await input[0].dataSync();
      expect(data.every(value => value >= 0 && value <= 255));
    });
    it('should respect flaot32 data range', async () => {
      const inputDef =
          [{shape: [1, 1, 10], dtype: 'float32', range: [0, 1], name: 'test'}];
      const input = generateInputFromDef(inputDef);
      expect(input.length).toEqual(1);
      expect(input[0]).toBeInstanceOf(tf.Tensor);
      expect(input[0].shape).toEqual([1, 1, 10]);
      expect(input[0].dtype).toEqual('float32');
      const data = await input[0].dataSync();
      expect(data.every(value => value >= 0 && value <= 1));
    });
  });

  describe('profile inference time', () => {
    describe('timeInference', () => {
      it('throws when passing in invalid predict', async () => {
        const predict = {};
        await expectAsync(timeInference(predict)).toBeRejected();
      });

      it('does not add new tensors', async () => {
        const model = tf.sequential(
            {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
        const input = tf.zeros([1, 3]);

        const tensorsBefore = tf.memory().numTensors;
        await timeInference(() => model.predict(input));
        expect(tf.memory().numTensors).toEqual(tensorsBefore);

        model.dispose();
        input.dispose();
      });
    });
  });

  describe('Profile Inference', () => {
    describe('profileInference', () => {
      it('pass in invalid predict', async () => {
        const predict = {};
        await expectAsync(profileInference(predict)).toBeRejected();
      });

      it('check tensor leak', async () => {
        const model = tf.sequential(
            {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
        const input = tf.zeros([1, 3]);

        const tensorsBefore = tf.memory().numTensors;
        await profileInference(() => model.predict(input));
        expect(tf.memory().numTensors).toEqual(tensorsBefore);

        model.dispose();
        input.dispose();
      });

      it('profile all statements in async predict function', async () => {
        const predict = async () => {
          await sleep(1);
          const x = tf.tensor1d([1, 2, 3]);
          const x2 = x.square();
          x2.dispose();
          return x;
        };

        const oldTensorCount = tf.memory().numTensors;
        let profileInfo = await profileInference(predict);
        expect(tf.memory().numTensors).toEqual(oldTensorCount);

        // If `profileInference` cannot profile async function, it would
        // fail to profile all the statements after `await sleep(1);` and the
        // peak memory would be `-Infinity`.
        expect(profileInfo.peakBytes).toBeGreaterThan(0);
      });
    });
  });

  describe('aggregateKernelTime', () => {
    it('aggregates the kernels according to names', () => {
      const kernels = [
        {name: 'testKernel1', kernelTimeMs: 1},
        {name: 'testKernel1', kernelTimeMs: 1},
        {name: 'testKernel1', kernelTimeMs: 1},
        {name: 'testKernel2', kernelTimeMs: 1},
        {name: 'testKernel2', kernelTimeMs: 1},
      ];

      const aggregatedKernels = aggregateKernelTime(kernels);
      expect(aggregatedKernels.length).toBe(2);
      expect(aggregatedKernels[0].name).toBe('testKernel1');
      expect(aggregatedKernels[0].timeMs).toBe(3);
      expect(aggregatedKernels[1].name).toBe('testKernel2');
      expect(aggregatedKernels[1].timeMs).toBe(2);
    });
  });

  describe('getPredictFnForModel', () => {
    it('graph model with async ops uses executeAsync to run', () => {
      const model = new tf.GraphModel();
      const input = tf.tensor([1]);
      const oldTensorNum = tf.memory().numTensors;
      spyOn(model, 'execute').and.callFake(() => {
        const leakedTensor = tf.tensor([1]);
        throw new Error(
            'This model has dynamic ops, ' +
            'please use model.executeAsync() instead');
        return leakedTensor;
      });
      spyOn(model, 'executeAsync');

      const wrappedPredict = getPredictFnForModel(model, input);
      expect(tf.memory().numTensors).toBe(oldTensorNum);
      expect(model.execute.calls.count()).toBe(1);
      expect(model.execute.calls.first().args).toEqual([input]);

      wrappedPredict();
      expect(model.execute.calls.count()).toBe(1);
      expect(model.executeAsync.calls.count()).toBe(1);
      expect(model.executeAsync.calls.first().args).toEqual([input]);

      tf.dispose(input);
    });

    it('graph model without async ops uses execute to run', () => {
      const model = new tf.GraphModel();
      const input = tf.tensor([1]);
      const oldTensorNum = tf.memory().numTensors;
      spyOn(model, 'execute').and.callFake(() => {
        const leakedTensor = tf.tensor([1]);
      });
      spyOn(model, 'executeAsync');

      const wrappedPredict = getPredictFnForModel(model, input);
      expect(tf.memory().numTensors).toBe(oldTensorNum);
      expect(model.execute.calls.count()).toBe(1);
      expect(model.execute.calls.first().args).toEqual([input]);

      wrappedPredict();
      expect(model.execute.calls.count()).toBe(2);
      expect(model.execute.calls.argsFor(1)).toEqual([input]);
      expect(model.executeAsync.calls.count()).toBe(0);

      tf.dispose(input);
    });

    it('layers model uses predict to run', () => {
      const model = tf.sequential(
          {layers: [tf.layers.dense({units: 1, inputShape: [1]})]});
      const input = tf.ones([1, 1]);
      spyOn(model, 'predict');

      const wrappedPredict = getPredictFnForModel(model, input);
      wrappedPredict();

      expect(model.predict.calls.count()).toBe(1);
      expect(model.predict.calls.first().args).toEqual([input]);

      tf.dispose(input);
      model.dispose();
    });

    it('throws when passed in a model that is not layers or graph model',
       () => {
         const model = {};
         const input = [];
         expect(() => getPredictFnForModel(model, input)).toThrowError(Error);
       });
  });

  describe('setEnvFlags', () => {
    describe('changes nothing when setting empty config or rejecting', () => {
      let originalFlags = {};

      beforeEach(() => {
        originalFlags = {...tf.env().flags};
      });
      afterAll(() => tf.env().reset());

      it('empty config', async () => {
        await setEnvFlags();
        expect(tf.env().flags).toEqual(originalFlags);
      });

      it('rejects when setting untunable flags', async () => {
        const flagConfig = {
          IS_BROWSER: false,
        };
        expectAsync(setEnvFlags(flagConfig))
            .toBeRejectedWithError(
                Error, /is not a tunable or valid environment flag./);
        expect(tf.env().flags).toEqual(originalFlags);
      });

      it('rejects when setting a number flag by a boolean value', async () => {
        const flagConfig = {
          WEBGL_VERSION: false,
        };
        expectAsync(setEnvFlags(flagConfig)).toBeRejectedWithError(Error);
        expect(tf.env().flags).toEqual(originalFlags);
      });

      it('rejects when setting boolean flag by a number', async () => {
        const flagConfig = {
          WEBGL_PACK: 1,
        };
        expectAsync(setEnvFlags(flagConfig)).toBeRejectedWithError(Error);
        expect(tf.env().flags).toEqual(originalFlags);
      });

      it('rejects when setting flag value out of the range', async () => {
        const outOfRangeValue =
            Math.max(...TUNABLE_FLAG_VALUE_RANGE_MAP.WEBGL_VERSION) + 1;
        const flagConfig = {
          WEBGL_VERSION: outOfRangeValue,
        };
        expectAsync(setEnvFlags(flagConfig)).toBeRejectedWithError(Error);
        expect(tf.env().flags).toEqual(originalFlags);
      });
    });

    describe('reset simple flags', () => {
      beforeEach(() => tf.env().reset());
      afterEach(() => tf.env().reset());

      it('reset number type flags', async () => {
        const flagConfig = {
          WEBGL_VERSION: 1,
        };
        await setEnvFlags(flagConfig);
        expect(tf.env().getNumber('WEBGL_VERSION')).toBe(1);
      });

      it('reset boolean flags', async () => {
        const flagConfig = {
          WASM_HAS_SIMD_SUPPORT: false,
          WEBGL_CPU_FORWARD: false,
          WEBGL_PACK: false,
          WEBGL_FORCE_F16_TEXTURES: false,
          WEBGL_RENDER_FLOAT32_CAPABLE: false,
        };
        await setEnvFlags(flagConfig);
        expect(tf.env().getBool('WASM_HAS_SIMD_SUPPORT')).toBe(false);
        expect(tf.env().getBool('WEBGL_CPU_FORWARD')).toBe(false);
        expect(tf.env().getBool('WEBGL_PACK')).toBe(false);
        expect(tf.env().getBool('WEBGL_FORCE_F16_TEXTURES')).toBe(false);
        expect(tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')).toBe(false);
      });
    });

    describe('reset flags related to environment initialization', () => {
      beforeEach(() => tf.engine().reset());
      afterAll(() => {
        tf.engine().reset();
        tf.setBackend('cpu');
      });

      it(`set 'WEBGL_VERSION' to 2`, async () => {
        if (!tf.webgl_util.isWebGLVersionEnabled(2)) {
          pending(
              'Please use a browser supporting WebGL 2.0 to run this test.');
        }
        const flagConfig = {
          WEBGL_VERSION: 2,
        };
        await setEnvFlags(flagConfig);
        expect(tf.env().getBool('WEBGL_BUFFER_SUPPORTED')).toBe(true);
      });

      it(`set 'WEBGL_VERSION' to 1`, async () => {
        if (!tf.webgl_util.isWebGLVersionEnabled(1)) {
          pending(
              'Please use a browser supporting WebGL 1.0 to run this test.');
        }
        const flagConfig = {
          WEBGL_VERSION: 1,
        };
        await setEnvFlags(flagConfig);
        expect(tf.env().getBool('WEBGL_BUFFER_SUPPORTED')).toBe(false);
      });

      it(`reset flags when the related backend is active`, async () => {
        if (!tf.webgl_util.isWebGLVersionEnabled(1)) {
          pending(
              'Please use a browser supporting WebGL 1.0 to run this test.');
        }
        await tf.setBackend('webgl');
        const flagConfig = {
          WEBGL_VERSION: 1,
        };
        await setEnvFlags(flagConfig);
        expect(tf.getBackend()).toBe('webgl');
      });

      it(`reset 'WASM_HAS_SIMD_SUPPORT' as true`,
         async () => {
             // TODO: add test for SIMD after SIMD implementation.
             // const simdSupported = await
             // env().getAsync('WASM_HAS_SIMD_SUPPORT');
         });

      it(`reset 'WASM_HAS_SIMD_SUPPORT' as false`, async () => {
        const flagConfig = {
          WASM_HAS_SIMD_SUPPORT: false,
        };
        await setEnvFlags(flagConfig);
        expect(tf.env().getBool('WASM_HAS_SIMD_SUPPORT')).toBe(false);
      });
    });
  });

  describe('resetBackend', () => {
    beforeEach(() => tf.setBackend('cpu'));
    afterAll(() => tf.engine().reset());

    it('rejects when resetting a backend that is not registed', async () => {
      expectAsync(resetBackend('invalidBackendName'))
          .toBeRejectedWithError(
              Error, 'invalidBackendName backend is not registed.');
    });

    it('do nothing when resetting a backend that is not created', async () => {
      const testCpuBackend = 'testCpuBackend';
      tf.registerBackend(testCpuBackend, tf.findBackendFactory('cpu'));
      expect(tf.engine().registry[testCpuBackend]).toBeUndefined();
      spyOn(tf, 'findBackendFactory');
      spyOn(tf, 'removeBackend');
      spyOn(tf, 'registerBackend');

      await resetBackend(testCpuBackend);

      expect(tf.findBackendFactory.calls.count()).toBe(0);
      expect(tf.removeBackend.calls.count()).toBe(0);
      expect(tf.registerBackend.calls.count()).toBe(0);
      tf.removeBackend(testCpuBackend);
    });

    it('reset the backend when resetting an existed backend', async () => {
      await tf.ready();
      const currentBackend = tf.getBackend();
      expect(tf.engine().registry[currentBackend]).toBeDefined();
      spyOn(tf, 'findBackendFactory');
      spyOn(tf, 'removeBackend');
      spyOn(tf, 'registerBackend');

      await resetBackend(currentBackend);

      expect(tf.findBackendFactory.calls.count()).toBe(1);
      expect(tf.removeBackend.calls.count()).toBe(1);
      expect(tf.registerBackend.calls.count()).toBe(1);
    });

    it('tf.setBackend is called when resetting the active backend',
       async () => {
         const currentBackend = tf.getBackend();
         spyOn(tf, 'setBackend');
         await resetBackend(currentBackend);
         expect(tf.setBackend.calls.count()).toBe(1);
       });

    it('tf.setBackend is not called when resetting an inactive backend',
       async () => {
         const testCpuBackend = 'testCpuBackend';
         tf.registerBackend(testCpuBackend, tf.findBackendFactory('cpu'));
         expect(tf.getBackend()).not.toBe(testCpuBackend);
         spyOn(tf, 'setBackend');

         await resetBackend(testCpuBackend);

         expect(tf.setBackend.calls.count()).toBe(0);
         tf.removeBackend(testCpuBackend);
       });
  });
});
