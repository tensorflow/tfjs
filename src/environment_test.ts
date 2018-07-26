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
import {describeWithFlags} from './jasmine_util';
import {KernelBackend} from './kernels/backend';
import {MathBackendCPU} from './kernels/backend_cpu';
import {MathBackendWebGL} from './kernels/backend_webgl';
import {WEBGL_ENVS} from './test_util';

describeWithFlags('disjoint query timer enabled', WEBGL_ENVS, () => {
  afterEach(() => {
    ENV.reset();
  });

  it('no webgl', () => {
    ENV.setFeatures({'WEBGL_VERSION': 0});
    expect(ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION')).toBe(0);
  });

  it('webgl 1', () => {
    const features: Features = {'WEBGL_VERSION': 1};

    spyOn(document, 'createElement').and.returnValue({
      getContext: (context: string) => {
        if (context === 'webgl' || context === 'experimental-webgl') {
          return {
            getExtension: (extensionName: string) => {
              if (extensionName === 'EXT_disjoint_timer_query') {
                return {};
              } else if (extensionName === 'WEBGL_lose_context') {
                return {loseContext: () => {}};
              }
              return null;
            }
          };
        }
        return null;
      }
    });

    ENV.setFeatures(features);
    // TODO(nsthorat): expect to be 1 when
    // https://github.com/tensorflow/tfjs/issues/544 is fixed.
    expect(ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION')).toBe(0);
  });

  it('webgl 2', () => {
    const features: Features = {'WEBGL_VERSION': 2};

    spyOn(document, 'createElement').and.returnValue({
      getContext: (context: string) => {
        if (context === 'webgl2') {
          return {
            getExtension: (extensionName: string) => {
              if (extensionName === 'EXT_disjoint_timer_query_webgl2') {
                return {};
              } else if (extensionName === 'WEBGL_lose_context') {
                return {loseContext: () => {}};
              }
              return null;
            }
          };
        }
        return null;
      }
    });

    ENV.setFeatures(features);
    // TODO(nsthorat): expect to be 2 when
    // https://github.com/tensorflow/tfjs/issues/544 is fixed.
    expect(ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION')).toBe(0);
  });
});

describeWithFlags(
    'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', WEBGL_ENVS, () => {
      afterEach(() => {
        ENV.reset();
      });

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

describeWithFlags('WEBGL_FENCE_API_ENABLED', WEBGL_ENVS, () => {
  afterEach(() => {
    ENV.reset();
  });

  beforeEach(() => {
    spyOn(document, 'createElement').and.returnValue({
      getContext: (context: string) => {
        if (context === 'webgl2') {
          return {
            getExtension: (extensionName: string) => {
              if (extensionName === 'WEBGL_get_buffer_sub_data_async') {
                return {};
              } else if (extensionName === 'WEBGL_lose_context') {
                return {loseContext: () => {}};
              }
              return null;
            },
            fenceSync: () => 1
          };
        }
        return null;
      }
    });
  });

  it('WebGL 2 enabled', () => {
    const features: Features = {'WEBGL_VERSION': 2};

    const env = new Environment(features);

    expect(env.get('WEBGL_FENCE_API_ENABLED')).toBe(true);
  });

  it('WebGL 1 disabled', () => {
    const features: Features = {'WEBGL_VERSION': 1};

    const env = new Environment(features);

    expect(env.get('WEBGL_FENCE_API_ENABLED')).toBe(false);
  });
});

describeWithFlags('WebGL version', WEBGL_ENVS, () => {
  afterEach(() => {
    ENV.reset();
  });

  it('webgl 1', () => {
    spyOn(document, 'createElement').and.returnValue({
      getContext: (context: string) => {
        if (context === 'webgl') {
          return {
            getExtension: (a: string) => {
              return {loseContext: () => {}};
            }
          };
        }
        return null;
      }
    });

    const env = new Environment();
    expect(env.get('WEBGL_VERSION')).toBe(1);
  });

  it('webgl 2', () => {
    spyOn(document, 'createElement').and.returnValue({
      getContext: (context: string) => {
        if (context === 'webgl2') {
          return {
            getExtension: (a: string) => {
              return {loseContext: () => {}};
            }
          };
        }
        return null;
      }
    });

    const env = new Environment();
    expect(env.get('WEBGL_VERSION')).toBe(2);
  });

  it('no webgl', () => {
    spyOn(document, 'createElement').and.returnValue({
      getContext: (context: string): WebGLRenderingContext => null
    });

    const env = new Environment();
    expect(env.get('WEBGL_VERSION')).toBe(0);
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
