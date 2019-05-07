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
    const env = new Environment({});
    env.setFlags({});

    const constraints = {};

    const backendName = 'test-backend';

    expect(envSatisfiesConstraints(env, backendName, constraints)).toBe(true);
  });

  it('ENV satisfies matching flag constraints no predicate', () => {
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});

    const constraints = {flags: {'TEST-FLAG': true}};

    const backendName = 'test-backend';

    expect(envSatisfiesConstraints(env, backendName, constraints)).toBe(true);
  });

  it('ENV satisfies matching flag and predicate is true', () => {
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});

    const constraints = {flags: {'TEST-FLAG': true}, predicate: () => true};

    const backendName = 'test-backend';

    expect(envSatisfiesConstraints(env, backendName, constraints)).toBe(true);
  });

  it('ENV doesnt satisfy flags and predicate is true', () => {
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});

    const constraints = {flags: {'TEST-FLAG': false}, predicate: () => true};

    const backendName = 'test-backend';

    expect(envSatisfiesConstraints(env, backendName, constraints)).toBe(false);
  });

  it('ENV satisfies flags and predicate is false', () => {
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});

    const constraints = {flags: {'TEST-FLAG': true}, predicate: () => false};

    const backendName = 'test-backend';

    expect(envSatisfiesConstraints(env, backendName, constraints)).toBe(false);
  });

  it('ENV doesnt satiisfy flags and predicate is false', () => {
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});

    const constraints = {flags: {'TEST-FLAG': false}, predicate: () => false};

    const backendName = 'test-backend';

    expect(envSatisfiesConstraints(env, backendName, constraints)).toBe(false);
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
