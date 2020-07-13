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
 * The unit tests in this file can be run by opening `SpecRunner.html` in
 * browser.
 */

describe('benchmark_util', () => {
  beforeAll(() => tf.setBackend('cpu'));

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

  describe('Profile Memory', () => {
    describe('profileInferenceMemory', () => {
      it('pass in invalid predict', async () => {
        const predict = {};
        await expectAsync(profileInferenceMemory(predict)).toBeRejected();
      });

      it('check tensor leak', async () => {
        const model = tf.sequential(
            {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
        const input = tf.zeros([1, 3]);

        const tensorsBefore = tf.memory().numTensors;
        await profileInferenceMemory(() => model.predict(input));
        expect(tf.memory().numTensors).toEqual(tensorsBefore);

        model.dispose();
        input.dispose();
      });
    });
  });

  describe('setEnvFlags', () => {
    describe('change nothing', () => {
      let originalFlags = {};

      beforeEach(() => {
        originalFlags = {...tf.env().flags};
      });
      afterAll(() => tf.env().reset());

      it('empty config', async () => {
        await setEnvFlags();
        expect(tf.env().flags).toEqual(originalFlags);
      });

      it('untunable flag', async () => {
        const flagConfig = {
          IS_BROWSER: false,
        };
        expectAsync(setEnvFlags(flagConfig))
            .toBeRejectedWithError(
                Error, /is not a tunable or valid environment flag./);
        expect(tf.env().flags).toEqual(originalFlags);
      });

      it('set a number type flag by a boolean value', async () => {
        const flagConfig = {
          WEBGL_VERSION: false,
        };
        expectAsync(setEnvFlags(flagConfig)).toBeRejectedWithError(Error);
        expect(tf.env().flags).toEqual(originalFlags);
      });

      it('set boolean flag by a number', async () => {
        const flagConfig = {
          WEBGL_PACK: 1,
        };
        expectAsync(setEnvFlags(flagConfig)).toBeRejectedWithError(Error);
        expect(tf.env().flags).toEqual(originalFlags);
      });
    });

    describe('reset flags', () => {
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

    it('reset a backend that is not registed', async () => {
      expectAsync(resetBackend('invalidBackendName'))
          .toBeRejectedWithError(
              Error, 'invalidBackendName backend is not registed.');
    });

    it('reset a backend that is not generated', async () => {
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

    it('reset a backend that has been generated', async () => {
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

    it('reset the active backend', async () => {
      const currentBackend = tf.getBackend();
      spyOn(tf, 'setBackend');
      await resetBackend(currentBackend);
      expect(tf.setBackend.calls.count()).toBe(1);
    });

    it('reset an inactive backend', async () => {
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
