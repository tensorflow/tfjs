
// import * as index from './index.js';

class testFolderController {
  __controllers = ['backendController'];
  add(childController) {
    this.__controllers.push(childController);
    return this;
  };
  remove(targetController) {
    const targetIndex = this.__controllers.indexOf(targetController);
    if (targetIndex >= 0) {
      this.__controllers.splice(targetIndex, 1);
    }
  }
}

const state = {
  flags: {}
};

describe('index', () => {
  describe('showFlagSettings', () => {
    // const BACKEND_FLAGS_MAP = {
    //   general: ['generalFlag0', 'generalFlag1'],
    //   backend0: ['flag00', 'flag01', 'flag02'],
    //   backend1: ['flag10', 'flag11', 'flag12', 'flag13']
    // };
    let oldInitDefaultValueMap;
    let oldShowBackendFlagSettings;

    beforeAll(() => {
      oldInitDefaultValueMap = initDefaultValueMap;
      oldShowBackendFlagSettings = showBackendFlagSettings;
    });

    beforeEach(() => {
      TUNABLE_FLAG_DEFAULT_VALUE_MAP = null;
    })

    afterAll(() => {
      initDefaultValueMap = oldInitDefaultValueMap;
      showBackendFlagSettings = oldShowBackendFlagSettings;
    });

    it('sets general flag settings when setting up from scratch', async () => {
      const folderController = {__controllers: ['backendController']};
      initDefaultValueMap = jasmine.createSpy();
      showBackendFlagSettings = jasmine.createSpy();
      // spyOn(index, 'initDefaultValueMap');
      // spyOn(index, 'showBackendFlagSettings');

      await showFlagSettings(folderController, 'webgl');

      expect(initDefaultValueMap.calls.count()).toBe(1);
      if (BACKEND_FLAGS_MAP.general.length > 0) {
        expect(showBackendFlagSettings.calls.count()).toBe(2);
        expect(showBackendFlagSettings.calls.argsFor(0)).toEqual([
          folderController, 'general'
        ]);
        expect(showBackendFlagSettings.calls.argsFor(1)).toEqual([
          folderController, 'webgl'
        ]);
      } else {
        expect(showBackendFlagSettings.calls.count()).toBe(1);
        expect(showBackendFlagSettings.calls.first().args).toEqual([
          folderController, 'webgl'
        ]);
      }
    });

    it('only sets settings for new backend if not from scratch', async () => {
      const folderController = {__controllers: ['backendController']};
      // Show general flag settings by create element controllers under the
      // parent controller.
      for (let index = 0; index < BACKEND_FLAGS_MAP.general.length; index++) {
        folderController.__controllers.push('generalFlagController');
      }
      initDefaultValueMap = jasmine.createSpy();
      showBackendFlagSettings = jasmine.createSpy();

      await showFlagSettings(folderController, 'webgl');

      expect(initDefaultValueMap.calls.count()).toBe(1);
      expect(showBackendFlagSettings.calls.count()).toBe(1);
      expect(showBackendFlagSettings.calls.first().args).toEqual([
        folderController, 'webgl'
      ]);
    });

    it('removes history flag settings except backend and general flag settings',
       async () => {
         const folderController = new testFolderController();
         // Show general flag settings.
         for (let index = 0; index < BACKEND_FLAGS_MAP.general.length;
              index++) {
           folderController.add(BACKEND_FLAGS_MAP.general[index]);
         }
         // Show webgl flag settings as the history flag settings.
         for (let index = 0; index < BACKEND_FLAGS_MAP.webgl.length; index++) {
           folderController.add(BACKEND_FLAGS_MAP.webgl[index]);
         }

         initDefaultValueMap = jasmine.createSpy();
         showBackendFlagSettings = jasmine.createSpy();
         spyOn(folderController, 'remove').and.callThrough();

         await showFlagSettings(folderController, 'wasm');

         expect(folderController.remove.calls.count())
             .toBe(BACKEND_FLAGS_MAP.webgl.length);
         // Show webgl flag settings as the history flag settings.
         for (let index = 0; index < BACKEND_FLAGS_MAP.webgl.length; index++) {
           expect(folderController.remove.calls.argsFor(index)).toEqual([
             BACKEND_FLAGS_MAP.webgl[index]
           ]);
         }
         expect(initDefaultValueMap.calls.count()).toBe(1);
         expect(showBackendFlagSettings.calls.count()).toBe(1);
         expect(showBackendFlagSettings.calls.first().args).toEqual([
           folderController, 'wasm'
         ]);
       });
  });

  describe('showBackendFlagSettings', () => {
    let oldGetTunableRange;

    beforeAll(() => {
      // Assume testBackend has only one flag, testFlag.
      // A DOM element is showed based on this flag's tunable range.
      BACKEND_FLAGS_MAP['testBackend'] = ['testFlag'];
      state.flags['testFlag'] = null;
      oldGetTunableRange = getTunableRange;
    });

    afterAll(() => {
      delete BACKEND_FLAGS_MAP['testBackend'];
      delete state.flags['testFlag'];
      getTunableRange = oldGetTunableRange;
    });

    it('does not show DOM element for untunable flags', () => {
      const folderController = new dat.gui.GUI();
      // The flag with only one value option is considered as untunable.
      const flagValueRange = [false];
      getTunableRange = jasmine.createSpy().and.returnValue(flagValueRange);
      spyOn(folderController, 'add');
      spyOn(console, 'warn');

      showBackendFlagSettings(folderController, 'testBackend');

      expect(folderController.add.calls.count()).toBe(0);
      expect(console.warn.calls.count()).toBe(1);

      folderController.destroy();
    });

    it('show checkbox for boolean tunable flags', () => {
      const folderController = new dat.gui.GUI();
      state.flags.testFlag = true;
      const flagValueRange = [true, false];
      getTunableRange = jasmine.createSpy().and.returnValue(flagValueRange);
      spyOn(folderController, 'add').and.callThrough();

      showBackendFlagSettings(folderController, 'testBackend');

      expect(folderController.add.calls.count()).toBe(1);
      expect(folderController.add.calls.first().args).toEqual([
        state.flags, 'testFlag'
      ]);
      expect(folderController.add.calls.first().returnValue.__checkbox)
          .toBeDefined();

      folderController.destroy();
    });

    it('show dropdown menu for number type tunable flags', () => {
      const folderController = new dat.gui.GUI();
      state.flags.testFlag = 1;
      const flagValueRange = [1, 2];
      getTunableRange = jasmine.createSpy().and.returnValue(flagValueRange);
      spyOn(folderController, 'add').and.callThrough();

      showBackendFlagSettings(folderController, 'testBackend');

      expect(folderController.add.calls.count()).toBe(1);
      expect(folderController.add.calls.first().args).toEqual([
        state.flags, 'testFlag', flagValueRange
      ]);
      expect(folderController.add.calls.first().returnValue.__select)
          .toBeDefined();

      folderController.destroy();
    });
  });

  describe('getTunableRange', () => {
    afterAll(() => {
      TUNABLE_FLAG_DEFAULT_VALUE_MAP = null;
    });

    it('returns [false] for the flag with false as default value', () => {
      TUNABLE_FLAG_DEFAULT_VALUE_MAP = {testFlag: false};
      const flagValueRange = getTunableRange('testFlag');
      expect(flagValueRange).toEqual([false]);
    });
    it('returns [false, true] for the flag with true as default value', () => {
      TUNABLE_FLAG_DEFAULT_VALUE_MAP = {testFlag: true};
      const flagValueRange = getTunableRange('testFlag');
      expect(flagValueRange).toEqual([false, true]);
    });
    it('returns [1..n] for the flag with number n as default value', () => {
      TUNABLE_FLAG_DEFAULT_VALUE_MAP = {testFlag: 5};
      const flagValueRange = getTunableRange('testFlag');
      expect(flagValueRange).toEqual([1, 2, 3, 4, 5]);
    });
  });
});
