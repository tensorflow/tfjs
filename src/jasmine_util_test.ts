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

import * as tf from './index';
import {describeWithFlags, envSatisfiesConstraints, parseKarmaFlags} from './jasmine_util';
import {MathBackendCPU} from './kernels/backend_cpu';
import {MathBackendWebGL} from './kernels/backend_webgl';
import {WEBGL_ENVS} from './test_util';

describe('jasmine_util.envSatisfiesConstraints', () => {
  it('ENV satisfies empty constraints', () => {
    expect(envSatisfiesConstraints({})).toBe(true);
  });

  it('ENV satisfies matching constraints', () => {
    const c = {TEST_EPSILON: tf.ENV.get('TEST_EPSILON')};
    expect(envSatisfiesConstraints(c)).toBe(true);
  });

  it('ENV does not satisfy mismatching constraints', () => {
    const c = {TEST_EPSILON: tf.ENV.get('TEST_EPSILON') + 0.1};
    expect(envSatisfiesConstraints(c)).toBe(false);
  });
});

describe('jasmine_util.parseKarmaFlags', () => {
  it('parse empty args', () => {
    const res = parseKarmaFlags([]);
    expect(res).toBeNull();
  });

  it('--backend cpu', () => {
    const res = parseKarmaFlags(['--backend', 'cpu']);
    expect(res.name).toBe('cpu');
    expect(res.features).toEqual({});
    expect(res.factory() instanceof MathBackendCPU).toBe(true);
  });

  it('--backend cpu --features {"IS_NODE": true}', () => {
    const res = parseKarmaFlags(
        ['--backend', 'cpu', '--features', '{"IS_NODE": true}']);
    expect(res.name).toBe('cpu');
    expect(res.features).toEqual({IS_NODE: true});
    expect(res.factory() instanceof MathBackendCPU).toBe(true);
  });

  it('"--backend unknown" throws error', () => {
    expect(() => parseKarmaFlags(['--backend', 'unknown'])).toThrowError();
  });

  it('"--features {}" throws error since --backend is missing', () => {
    expect(() => parseKarmaFlags(['--features', '{}'])).toThrowError();
  });

  it('"--backend cpu --features" throws error since features value is missing',
     () => {
       expect(() => parseKarmaFlags(['--backend', 'cpu', '--features']))
           .toThrowError();
     });

  it('"--backend cpu --features notJson" throws error', () => {
    expect(() => parseKarmaFlags(['--backend', 'cpu', '--features', 'notJson']))
        .toThrowError();
  });
});

describeWithFlags('jasmine_util.envSatisfiesConstraints', WEBGL_ENVS, () => {
  it('--backend webgl', () => {
    const res = parseKarmaFlags(['--backend', 'webgl']);
    expect(res.name).toBe('webgl');
    expect(res.features).toEqual({});
    expect(res.factory() instanceof MathBackendWebGL).toBe(true);
  });
});
