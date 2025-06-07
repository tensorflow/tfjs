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

const state = {
  backend: 'wasm',
  flags: {}
};

describe('index', () => {
  describe('showFlagSettings', () => {
    beforeAll(() => {
      this.originalInitDefaultValueMap = initDefaultValueMap;
      this.originalShowBackendFlagSettings = showBackendFlagSettings;
      this.originalTUNABLE_FLAG_DEFAULT_VALUE_MAP =
          TUNABLE_FLAG_DEFAULT_VALUE_MAP;
    });

    afterAll(() => {
      initDefaultValueMap = this.originalInitDefaultValueMap;
      showBackendFlagSettings = this.originalShowBackendFlagSettings;
      TUNABLE_FLAG_DEFAULT_VALUE_MAP =
          this.originalTUNABLE_FLAG_DEFAULT_VALUE_MAP;
    });

    describe('at the first call', () => {
      beforeAll(() => {
        // Only the backend setting is shown and the general flag settings have
        // not been shown.
        this.folderController = new dat.gui.GUI();
        this.folderController.add(state, 'backend', ['wasm', 'webgl', 'cpu']);

        // The flag default value map has not been initialized.
        TUNABLE_FLAG_DEFAULT_VALUE_MAP = null;

        BACKEND_FLAGS_MAP.general.push('testGeneralFlag');
      });

      afterAll(() => {
        this.folderController.destroy();
        BACKEND_FLAGS_MAP.general.pop();  // Pop 'testGeneralFlag'.
      });

      it('shows general flag settings', async () => {
        initDefaultValueMap = jasmine.createSpy();
        showBackendFlagSettings = jasmine.createSpy();

        await showFlagSettings(this.folderController, 'webgl');

        expect(showBackendFlagSettings.calls.count()).toBe(2);
        expect(showBackendFlagSettings.calls.first().args).toEqual([
          this.folderController, 'general'
        ]);
      });

      it('initializes default value map', async () => {
        initDefaultValueMap = jasmine.createSpy();
        showBackendFlagSettings = jasmine.createSpy();
        await showFlagSettings(this.folderController, 'webgl');
        expect(initDefaultValueMap.calls.count()).toBe(1);
      });

      it('shows flag settings for the given backend', async () => {
        initDefaultValueMap = jasmine.createSpy();
        showBackendFlagSettings = jasmine.createSpy();

        await showFlagSettings(this.folderController, 'webgl');

        expect(showBackendFlagSettings.calls.count()).toBe(2);
        expect(showBackendFlagSettings.calls.argsFor(0)).toEqual([
          this.folderController, 'general'
        ]);
        expect(showBackendFlagSettings.calls.argsFor(1)).toEqual([
          this.folderController, 'webgl'
        ]);
      });
    });

    describe('When switching the backend', () => {
      beforeAll(() => {
        // The flag default value map has been initialized and the state.flags
        // is populated with all tunable flags.
        this.originalInitDefaultValueMap();
        BACKEND_FLAGS_MAP.general.push('testGeneralFlag');
        TUNABLE_FLAG_DEFAULT_VALUE_MAP.testGeneralFlag = true;
        state.flags.testGeneralFlag = true;
      });

      afterAll(() => {
        BACKEND_FLAGS_MAP.general.pop();  // Pop 'testGeneralFlag'.
        delete TUNABLE_FLAG_DEFAULT_VALUE_MAP['testGeneralFlag'];
        delete state.flags['testGeneralFlag'];
      });

      beforeEach(() => {
        // The backend setting and the general flag settings have been shown.
        this.folderController = new dat.gui.GUI();
        this.folderController.add(state, 'backend', ['wasm', 'webgl', 'cpu']);
        this.originalShowBackendFlagSettings(this.folderController, 'general');

        // Flag settings for a certain backend have been shown.
        this.originalShowBackendFlagSettings(this.folderController, 'webgl');
      });

      afterEach(() => {
        this.folderController.destroy();
      });

      it('removes flag settings of the previous backend', async () => {
        initDefaultValueMap = jasmine.createSpy();
        showBackendFlagSettings = jasmine.createSpy();
        spyOn(this.folderController, 'remove').and.callThrough();

        await showFlagSettings(this.folderController, 'wasm');

        expect(initDefaultValueMap.calls.count()).toBe(0);
        expect(this.folderController.remove.calls.count())
            .toBe(BACKEND_FLAGS_MAP.webgl.length);
      });

      it('only add flag settings for the new backend', async () => {
        initDefaultValueMap = jasmine.createSpy();
        showBackendFlagSettings = jasmine.createSpy();

        await showFlagSettings(folderController, 'webgl');

        expect(initDefaultValueMap.calls.count()).toBe(0);
        expect(showBackendFlagSettings.calls.count()).toBe(1);
        expect(showBackendFlagSettings.calls.first().args).toEqual([
          this.folderController, 'webgl'
        ]);
      });
    });
  });

  describe('showBackendFlagSettings', () => {
    beforeAll(() => {
      // Assume testBackend has only one flag, testFlag.
      // A DOM element is shown based on this flag's tunable range.
      BACKEND_FLAGS_MAP['testBackend'] = ['testFlag'];
      state.flags['testFlag'] = null;
      this.originalGetTunableRange = getTunableRange;

      this.folderController = new dat.gui.GUI();
    });

    afterAll(() => {
      delete BACKEND_FLAGS_MAP['testBackend'];
      delete state.flags['testFlag'];
      getTunableRange = this.originalGetTunableRange;

      this.folderController.destroy();
    });

    it('does not show DOM element for untunable flags', () => {
      // The flag with only one value option is considered as untunable.
      const flagValueRange = [false];
      getTunableRange = jasmine.createSpy().and.returnValue(flagValueRange);
      spyOn(this.folderController, 'add');
      spyOn(console, 'warn');

      showBackendFlagSettings(this.folderController, 'testBackend');

      expect(this.folderController.add.calls.count()).toBe(0);
      expect(console.warn.calls.count()).toBe(1);
    });

    it('shows a checkbox for a tunable boolean flag', () => {
      state.flags.testFlag = true;
      const flagValueRange = [true, false];
      getTunableRange = jasmine.createSpy().and.returnValue(flagValueRange);
      spyOn(this.folderController, 'add').and.callThrough();

      showBackendFlagSettings(this.folderController, 'testBackend');

      expect(this.folderController.add.calls.count()).toBe(1);
      expect(this.folderController.add.calls.first().args).toEqual([
        state.flags, 'testFlag'
      ]);
      expect(this.folderController.add.calls.first().returnValue.__checkbox)
          .toBeDefined();
    });

    it('shows a dropdown menu for a tunable number type flag', () => {
      state.flags.testFlag = 1;
      const flagValueRange = [1, 2];
      getTunableRange = jasmine.createSpy().and.returnValue(flagValueRange);
      spyOn(this.folderController, 'add').and.callThrough();

      showBackendFlagSettings(this.folderController, 'testBackend');

      expect(this.folderController.add.calls.count()).toBe(1);
      expect(this.folderController.add.calls.first().args).toEqual([
        state.flags, 'testFlag', flagValueRange
      ]);
      expect(this.folderController.add.calls.first().returnValue.__select)
          .toBeDefined();
    });
  });

  describe('getTunableRange', () => {
    beforeAll(() => {
      this.originalTUNABLE_FLAG_DEFAULT_VALUE_MAP =
          TUNABLE_FLAG_DEFAULT_VALUE_MAP;
      TUNABLE_FLAG_DEFAULT_VALUE_MAP = {};
    });

    afterAll(() => {
      TUNABLE_FLAG_DEFAULT_VALUE_MAP =
          this.originalTUNABLE_FLAG_DEFAULT_VALUE_MAP;
    });

    it(`returns [false, true] for 'WEBGL_FORCE_F16_TEXTURES'`, () => {
      const flagValueRange = getTunableRange('WEBGL_FORCE_F16_TEXTURES');
      expect(flagValueRange).toEqual([false, true]);
    });

    it(`returns [1, 2] for 'WEBGL_VERSION' if its default value is 2`, () => {
      TUNABLE_FLAG_DEFAULT_VALUE_MAP.WEBGL_VERSION = 2;
      const flagValueRange = getTunableRange('WEBGL_VERSION');
      expect(flagValueRange).toEqual([1, 2]);
    });

    it(`returns [1] for 'WEBGL_VERSION' if its default value is 1`, () => {
      TUNABLE_FLAG_DEFAULT_VALUE_MAP.WEBGL_VERSION = 1;
      const flagValueRange = getTunableRange('WEBGL_VERSION');
      expect(flagValueRange).toEqual([1]);
    });

    it('returns [false] for the flag with false as default value', () => {
      TUNABLE_FLAG_DEFAULT_VALUE_MAP.testFlag = false;
      const flagValueRange = getTunableRange('testFlag');
      expect(flagValueRange).toEqual([false]);
    });

    it('returns [false, true] for the flag with true as default value', () => {
      TUNABLE_FLAG_DEFAULT_VALUE_MAP.testFlag = true;
      const flagValueRange = getTunableRange('testFlag');
      expect(flagValueRange).toEqual([false, true]);
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
});
