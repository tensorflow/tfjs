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
import {envSatisfiesConstraints, parseTestEnvFromKarmaFlags, TestEnv} from './jasmine_util';

describe('jasmine_util.envSatisfiesConstraints', () => {
  it('ENV satisfies empty constraints', () => {
    const env = new Environment({});
    env.setFlags({});

    const constraints = {};

    const backendName = 'test-backend';

    expect(
        envSatisfiesConstraints(env, {name: 'test', backendName}, constraints))
        .toBe(true);
  });

  it('ENV satisfies matching flag constraints no predicate', () => {
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});

    const constraints = {flags: {'TEST-FLAG': true}};

    const backendName = 'test-backend';

    expect(
        envSatisfiesConstraints(env, {name: 'test', backendName}, constraints))
        .toBe(true);
  });

  it('ENV satisfies matching flag and predicate is true', () => {
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});

    const constraints = {flags: {'TEST-FLAG': true}, predicate: () => true};

    const backendName = 'test-backend';

    expect(
        envSatisfiesConstraints(env, {name: 'test', backendName}, constraints))
        .toBe(true);
  });

  it('ENV doesnt satisfy flags and predicate is true', () => {
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});

    const constraints = {flags: {'TEST-FLAG': false}, predicate: () => true};

    const backendName = 'test-backend';

    expect(
        envSatisfiesConstraints(env, {name: 'test', backendName}, constraints))
        .toBe(false);
  });

  it('ENV satisfies flags and predicate is false', () => {
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});

    const constraints = {flags: {'TEST-FLAG': true}, predicate: () => false};

    const backendName = 'test-backend';

    expect(
        envSatisfiesConstraints(env, {name: 'test', backendName}, constraints))
        .toBe(false);
  });

  it('ENV doesnt satiisfy flags and predicate is false', () => {
    const env = new Environment({});
    env.setFlags({'TEST-FLAG': true});

    const constraints = {flags: {'TEST-FLAG': false}, predicate: () => false};

    const backendName = 'test-backend';

    expect(
        envSatisfiesConstraints(env, {name: 'test', backendName}, constraints))
        .toBe(false);
  });
});

describe('jasmine_util.parseKarmaFlags', () => {
  const registeredTestEnvs: TestEnv[] = [
    {name: 'test-env', backendName: 'test-backend', isDataSync: true, flags: {}}
  ];

  it('parse empty args', () => {
    const res = parseTestEnvFromKarmaFlags([], registeredTestEnvs);
    expect(res).toBeNull();
  });

  it('--testEnv test-env --flags {"IS_NODE": true}', () => {
    const res = parseTestEnvFromKarmaFlags(
        ['--testEnv', 'test-env', '--flags', '{"IS_NODE": true}'],
        registeredTestEnvs);
    expect(res.name).toBe('test-env');
    expect(res.backendName).toBe('test-backend');
    expect(res.flags).toEqual({IS_NODE: true});
  });

  it('"--testEnv unknown" throws error', () => {
    expect(
        () => parseTestEnvFromKarmaFlags(
            ['--testEnv', 'unknown'], registeredTestEnvs))
        .toThrowError();
  });

  it('"--flags {}" throws error since --testEnv is missing', () => {
    expect(
        () => parseTestEnvFromKarmaFlags(['--flags', '{}'], registeredTestEnvs))
        .toThrowError();
  });

  it('"--testEnv cpu --flags" throws error since features value is missing',
     () => {
       expect(
           () => parseTestEnvFromKarmaFlags(
               ['--testEnv', 'test-env', '--flags'], registeredTestEnvs))
           .toThrowError();
     });

  it('"--backend cpu --flags notJson" throws error', () => {
    expect(
        () => parseTestEnvFromKarmaFlags(
            ['--testEnv', 'test-env', '--flags', 'notJson'],
            registeredTestEnvs))
        .toThrowError();
  });
});
