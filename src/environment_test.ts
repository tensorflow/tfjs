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

import * as device_util from './device_util';
import {ENV, Environment} from './environment';
import {Features, getQueryParams} from './environment_util';
import * as tf from './index';
import {describeWithFlags} from './jasmine_util';
import {KernelBackend} from './kernels/backend';
import {MathBackendCPU} from './kernels/backend_cpu';
import {MathBackendWebGL} from './kernels/backend_webgl';
import {ALL_ENVS, expectArraysClose, WEBGL_ENVS} from './test_util';

describeWithFlags(
    'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', WEBGL_ENVS, () => {
      it('disjoint query timer disabled', () => {
        const features:
            Features = {'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION': 0};

        const env = new Environment(features);

        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
            .toBe(false);
      });

      it('disjoint query timer enabled, mobile', () => {
        const features:
            Features = {'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION': 1};
        spyOn(device_util, 'isMobile').and.returnValue(true);

        const env = new Environment(features);

        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
            .toBe(false);
      });

      it('disjoint query timer enabled, not mobile', () => {
        const features:
            Features = {'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION': 1};
        spyOn(device_util, 'isMobile').and.returnValue(false);

        const env = new Environment(features);

        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
            .toBe(true);
      });
    });

describeWithFlags('WEBGL_PAGING_ENABLED', WEBGL_ENVS, testEnv => {
  afterEach(() => {
    ENV.reset();
    ENV.setFeatures(testEnv.features);
  });

  it('should be true if in a browser', () => {
    const features: Features = {'IS_BROWSER': true};
    const env = new Environment(features);
    expect(env.get('WEBGL_PAGING_ENABLED')).toBe(true);
  });

  it('should not cause errors when paging is turned off', () => {
    ENV.set('WEBGL_PAGING_ENABLED', false);

    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);

    const c = tf.matMul(a, b);

    expectArraysClose(c, [0, 8, -3, 20]);
  });

  it('should be false when the environment is prod', () => {
    const features: Features = {'IS_BROWSER': true};
    const env = new Environment(features);
    env.set('PROD', true);
    expect(env.get('WEBGL_PAGING_ENABLED')).toBe(false);
  });
});

describe('Backend', () => {
  beforeAll(() => {
    // Silences backend registration warnings.
    spyOn(console, 'warn');
  });

  afterEach(() => {
    ENV.reset();
  });

  it('custom cpu registration', () => {
    let backend: KernelBackend;
    ENV.registerBackend('custom-cpu', () => {
      backend = new MathBackendCPU();
      return backend;
    });

    expect(ENV.findBackend('custom-cpu')).toBe(backend);
    Environment.setBackend('custom-cpu');
    expect(ENV.backend).toBe(backend);

    ENV.removeBackend('custom-cpu');
  });

  it('webgl not supported, falls back to cpu', () => {
    ENV.setFeatures({'WEBGL_VERSION': 0});
    let cpuBackend: KernelBackend;
    ENV.registerBackend('custom-cpu', () => {
      cpuBackend = new MathBackendCPU();
      return cpuBackend;
    }, 103);
    const success =
        ENV.registerBackend('custom-webgl', () => new MathBackendWebGL(), 104);
    expect(success).toBe(false);
    expect(ENV.findBackend('custom-webgl') == null).toBe(true);
    expect(Environment.getBackend()).toBe('custom-cpu');
    expect(ENV.backend).toBe(cpuBackend);

    ENV.removeBackend('custom-cpu');
  });

  it('default custom background null', () => {
    expect(ENV.findBackend('custom')).toBeNull();
  });

  it('allow custom backend', () => {
    const backend = new MathBackendCPU();
    const success = ENV.registerBackend('custom', () => backend);
    expect(success).toBeTruthy();
    expect(ENV.findBackend('custom')).toEqual(backend);
    ENV.removeBackend('custom');
  });
});

describe('environment_util.getQueryParams', () => {
  it('basic', () => {
    expect(getQueryParams('?a=1&b=hi&f=animal'))
        .toEqual({'a': '1', 'b': 'hi', 'f': 'animal'});
  });
});

describeWithFlags('max texture size', WEBGL_ENVS, () => {
  it('should not throw exception', () => {
    expect(() => ENV.get('WEBGL_MAX_TEXTURE_SIZE')).not.toThrow();
  });
});

describeWithFlags('epsilon', {}, () => {
  it('Epsilon is a function of float precision', () => {
    const epsilonValue = ENV.backend.floatPrecision() === 32 ? 1e-7 : 1e-3;
    expect(ENV.get('EPSILON')).toBe(epsilonValue);
  });

  it('abs(epsilon) > 0', () => {
    expect(tf.abs(ENV.get('EPSILON')).get()).toBeGreaterThan(0);
  });
});

describeWithFlags('TENSORLIKE_CHECK_SHAPE_CONSISTENCY', ALL_ENVS, () => {
  it('disabled when prod is enabled', () => {
    const env = new Environment();
    env.set('PROD', true);
    expect(env.get('TENSORLIKE_CHECK_SHAPE_CONSISTENCY')).toBe(false);
  });

  it('enabled when prod is disabled', () => {
    const env = new Environment();
    env.set('PROD', false);
    expect(env.get('TENSORLIKE_CHECK_SHAPE_CONSISTENCY')).toBe(true);
  });
});

describeWithFlags('WEBGL_SIZE_UPLOAD_UNIFORM', WEBGL_ENVS, () => {
  it('is 0 when there is no float32 bit support', () => {
    const env = new Environment();
    env.set('WEBGL_RENDER_FLOAT32_ENABLED', false);
    expect(env.get('WEBGL_SIZE_UPLOAD_UNIFORM')).toBe(0);
  });

  it('is > 0 when there is float32 bit support', () => {
    const env = new Environment();
    env.set('WEBGL_RENDER_FLOAT32_ENABLED', true);
    expect(env.get('WEBGL_SIZE_UPLOAD_UNIFORM')).toBeGreaterThan(0);
  });
});
