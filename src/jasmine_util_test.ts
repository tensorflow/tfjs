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

import {Environment} from './environment';
import * as tf from './index';
import {envSatisfiesConstraints, parseKarmaFlags, TestKernelBackend} from './jasmine_util';

describe('jasmine_util.envSatisfiesConstraints', () => {
  it('ENV satisfies empty constraints', () => {
    const backendName = 'test-backend';
    const env = new Environment({});
    env.setFlags({});
    const registeredBackends = ['test-backend1', 'test-backend2'];

    const constraints = {};

    expect(envSatisfiesConstraints(
               env, backendName, registeredBackends, constraints))
        .toBe(true);
  });

  it('ENV satisfies matching flag constraints, no backend constraint', () => {
    const backendName = 'test-backend';
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});
    const registeredBackends = ['test-backend1', 'test-backend2'];

    const constraints = {flags: {'TEST-FLAG': true}};

    expect(envSatisfiesConstraints(
               env, backendName, registeredBackends, constraints))
        .toBe(true);
  });

  it('ENV satisfies matching flag and one backend constraint', () => {
    const backendName = 'test-backend';
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});
    const registeredBackends = ['test-backend1', 'test-backend2'];

    const constraints = {flags: {'TEST-FLAG': true}, backends: backendName};

    expect(envSatisfiesConstraints(
               env, backendName, registeredBackends, constraints))
        .toBe(true);
  });

  it('ENV satisfies matching flag and multiple backend constraints', () => {
    const backendName = 'test-backend';
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});
    const registeredBackends = ['test-backend1', 'test-backend2'];

    const constraints = {
      flags: {'TEST-FLAG': true},
      backends: [backendName, 'other-backend']
    };

    expect(envSatisfiesConstraints(
               env, backendName, registeredBackends, constraints))
        .toBe(true);
  });

  it('ENV does not satisfy mismatching flags constraints', () => {
    const backendName = 'test-backend';
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': false});
    const registeredBackends = ['test-backend1', 'test-backend2'];

    const constraints = {flags: {'TEST-FLAG': true}};

    expect(envSatisfiesConstraints(
               env, backendName, registeredBackends, constraints))
        .toBe(false);
  });

  it('ENV satisfies no flag constraint but not satisfy activebackend', () => {
    const backendName = 'test-backend';
    const env = new Environment({});
    const registeredBackends = ['test-backend1', 'test-backend2'];

    const constraints = {activeBackend: 'test-backend2'};

    expect(envSatisfiesConstraints(
               env, backendName, registeredBackends, constraints))
        .toBe(false);
  });

  it('ENV satisfies flags but does not satisfy active backend', () => {
    const backendName = 'test-backend';
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});
    const registeredBackends = ['test-backend1', 'test-backend2'];

    const constraints = {
      flags: {'TEST-FLAG': true},
      activeBackend: 'test-backend2'
    };

    expect(envSatisfiesConstraints(
               env, backendName, registeredBackends, constraints))
        .toBe(false);
  });

  it('ENV satisfies flags active backend, but not registered backends', () => {
    const backendName = 'test-backend';
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});
    const registeredBackends = ['test-backend1'];

    const constraints = {
      flags: {'TEST-FLAG': true},
      activeBackend: 'test-backend1',
      registeredBackends: ['test-backend1', 'test-backend2']
    };

    expect(envSatisfiesConstraints(
               env, backendName, registeredBackends, constraints))
        .toBe(false);
  });
});

describe('jasmine_util.parseKarmaFlags', () => {
  it('parse empty args', () => {
    const res = parseKarmaFlags([]);
    expect(res).toBeNull();
  });

  it('--backend test-backend --flags {"IS_NODE": true}', () => {
    const backend = new TestKernelBackend();
    tf.registerBackend('test-backend', () => backend);

    const res = parseKarmaFlags(
        ['--backend', 'test-backend', '--flags', '{"IS_NODE": true}']);
    expect(res.name).toBe('test-backend');
    expect(res.backendName).toBe('test-backend');
    expect(res.flags).toEqual({IS_NODE: true});

    tf.removeBackend('test-backend');
  });

  it('"--backend unknown" throws error', () => {
    expect(() => parseKarmaFlags(['--backend', 'unknown'])).toThrowError();
  });

  it('"--flags {}" throws error since --backend is missing', () => {
    expect(() => parseKarmaFlags(['--flags', '{}'])).toThrowError();
  });

  it('"--backend cpu --flags" throws error since features value is missing',
     () => {
       expect(() => parseKarmaFlags(['--backend', 'cpu', '--flags']))
           .toThrowError();
     });

  it('"--backend cpu --flags notJson" throws error', () => {
    expect(() => parseKarmaFlags(['--backend', 'cpu', '--flags', 'notJson']))
        .toThrowError();
  });
});
