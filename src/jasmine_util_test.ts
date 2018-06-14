/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {ENV, Environment, Features} from './environment';
import * as jasmine_util from './jasmine_util';
import {MathBackendCPU} from './kernels/backend_cpu';
import {NATIVE_ENV} from './test_util';
import {MathBackendWebGL} from './webgl';

describe('canEmulateEnvironment', () => {
  beforeEach(() => {
    ENV.reset();
  });
  afterEach(() => {
    ENV.reset();
  });

  it('no registered backends', () => {
    const fakeFeatures = {'BACKEND': 'fake-webgl'};

    expect(jasmine_util.canEmulateEnvironment(fakeFeatures)).toBe(false);
  });

  it('webgl backend, webgl emulation', () => {
    ENV.registerBackend('fake-webgl', () => new MathBackendCPU());

    const fakeFeatures = {'BACKEND': 'fake-webgl'};
    expect(jasmine_util.canEmulateEnvironment(fakeFeatures)).toBe(true);

    ENV.removeBackend('fake-webgl');
  });

  it('webgl backend, tensorflow emulation', () => {
    ENV.registerBackend('fake-webgl', () => new MathBackendCPU());

    const fakeFeatures = {'BACKEND': 'fake-tensorflow'};
    expect(jasmine_util.canEmulateEnvironment(fakeFeatures)).toBe(false);

    ENV.removeBackend('fake-webgl');
  });

  it('webgl backend, webgl 2.0 emulation on webgl 2.0', () => {
    ENV.registerBackend('fake-webgl', () => new MathBackendCPU());
    ENV.set('WEBGL_VERSION', 2);

    const fakeFeatures = {'BACKEND': 'fake-webgl', 'WEBGL_VERSION': 2};
    expect(jasmine_util.canEmulateEnvironment(fakeFeatures)).toBe(true);

    ENV.removeBackend('fake-webgl');
  });

  it('webgl backend, webgl 1.0 emulation on webgl 2.0', () => {
    ENV.registerBackend('fake-webgl', () => new MathBackendCPU());
    ENV.set('WEBGL_VERSION', 2);

    const fakeFeatures = {'BACKEND': 'fake-webgl', 'WEBGL_VERSION': 1};
    expect(jasmine_util.canEmulateEnvironment(fakeFeatures)).toBe(true);
    ENV.removeBackend('fake-webgl');
  });

  it('webgl backend, webgl 2.0 emulation on webgl 1.0 fails', () => {
    ENV.registerBackend('fake-webgl', () => new MathBackendCPU());
    ENV.set('WEBGL_VERSION', 1);

    const fakeFeatures = {'BACKEND': 'fake-webgl', 'WEBGL_VERSION': 2};
    expect(jasmine_util.canEmulateEnvironment(fakeFeatures)).toBe(false);

    ENV.removeBackend('fake-webgl');
  });

  it('webgl backend, webgl 1.0 no float emulation on webgl 2.0', () => {
    ENV.registerBackend('fake-webgl', () => new MathBackendCPU());
    ENV.set('WEBGL_VERSION', 2);
    ENV.set('WEBGL_RENDER_FLOAT32_ENABLED', true);

    // Emulates iOS.
    const fakeFeatures = {
      'BACKEND': 'fake-webgl',
      'WEBGL_VERSION': 1,
      'WEBGL_RENDER_FLOAT32_ENABLED': false
    };
    expect(jasmine_util.canEmulateEnvironment(fakeFeatures)).toBe(true);

    ENV.removeBackend('fake-webgl');
  });

  it('webgl backend, webgl 1.0 no float emulation on webgl 1.0 no float',
     () => {
       ENV.registerBackend('fake-webgl', () => new MathBackendCPU());
       ENV.set('WEBGL_VERSION', 1);
       ENV.set('WEBGL_RENDER_FLOAT32_ENABLED', false);
       ENV.set('WEBGL_DOWNLOAD_FLOAT_ENABLED', false);

       // Emulates iOS.
       const fakeFeatures = {
         'BACKEND': 'fake-webgl',
         'WEBGL_VERSION': 1,
         'WEBGL_RENDER_FLOAT32_ENABLED': false,
         'WEBGL_DOWNLOAD_FLOAT_ENABLED': false
       };
       expect(jasmine_util.canEmulateEnvironment(fakeFeatures)).toBe(true);

       ENV.removeBackend('fake-webgl');
     });
});

describe('anyFeaturesEquivalentToDefault', () => {
  let testBackends: jasmine_util.TestBackendFactory[];
  beforeEach(() => {
    testBackends = jasmine_util.TEST_BACKENDS;
  });
  afterEach(() => {
    jasmine_util.setTestBackends(testBackends);
  });

  it('ignores default', () => {
    const env = new Environment();
    const features = [NATIVE_ENV];
    expect(jasmine_util.anyFeaturesEquivalentToDefault(features, env))
        .toBe(false);
  });

  it('equivalent features', () => {
    jasmine_util.setTestBackends([{
      name: 'fake-webgl',
      factory: () => new MathBackendWebGL(),
      priority: 1000
    }]);

    const env = new Environment();
    env.set('WEBGL_VERSION', 1);
    env.set('BACKEND', 'fake-webgl');

    const features: Features[] =
        [NATIVE_ENV, {'WEBGL_VERSION': 1, 'BACKEND': 'fake-webgl'}];
    expect(jasmine_util.anyFeaturesEquivalentToDefault(features, env))
        .toBe(true);
  });

  it('different features', () => {
    jasmine_util.setTestBackends([{
      name: 'fake-webgl',
      factory: () => new MathBackendWebGL(),
      priority: 1
    }]);

    const env = new Environment();
    env.set('WEBGL_VERSION', 0);
    env.set('BACKEND', 'fake-cpu');

    const features: Features[] =
        [NATIVE_ENV].concat([{'WEBGL_VERSION': 1, 'BACKEND': 'fake-webgl'}]);
    expect(jasmine_util.anyFeaturesEquivalentToDefault(features, env))
        .toBe(false);
  });
});
