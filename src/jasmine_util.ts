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
import {KernelBackend} from './backends/backend';
import {ENGINE} from './engine';
import {ENV, Environment, Flags} from './environment';

Error.stackTraceLimit = Infinity;

export type Constraints = {
  flags?: Flags;
  activeBackend?: string;
  // If defined, all backends in this array must be registered.
  registeredBackends?: string[];
};

export const NODE_ENVS: Constraints = {
  flags: {'IS_NODE': true}
};
export const CHROME_ENVS: Constraints = {
  flags: {'IS_CHROME': true}
};
export const BROWSER_ENVS: Constraints = {
  flags: {'IS_BROWSER': true}
};

export const ALL_ENVS: Constraints = {};

// Tests whether the current environment satisfies the set of constraints.
export function envSatisfiesConstraints(
    env: Environment, activeBackend: string, registeredBackends: string[],
    constraints: Constraints): boolean {
  if (constraints == null) {
    return true;
  }

  if (constraints.flags != null) {
    for (const flagName in constraints.flags) {
      const flagValue = constraints.flags[flagName];
      if (env.get(flagName) !== flagValue) {
        return false;
      }
    }
  }
  if (constraints.activeBackend != null) {
    return activeBackend === constraints.activeBackend;
  }
  if (constraints.registeredBackends != null) {
    for (let i = 0; i < constraints.registeredBackends.length; i++) {
      const registeredBackendConstraint = constraints.registeredBackends[i];
      if (registeredBackends.indexOf(registeredBackendConstraint) === -1) {
        return false;
      }
    }
  }

  return true;
}

// tslint:disable-next-line:no-any
declare let __karma__: any;

export function parseKarmaFlags(args: string[]): TestEnv {
  let flags: Flags;
  let factory: () => KernelBackend;
  let backendName = '';
  const backendNames = ENGINE.backendNames()
                           .map(backendName => '\'' + backendName + '\'')
                           .join(', ');

  args.forEach((arg, i) => {
    if (arg === '--flags') {
      flags = JSON.parse(args[i + 1]);
    } else if (arg === '--backend') {
      const type = args[i + 1];
      backendName = type;
      factory = ENGINE.findBackendFactory(backendName.toLowerCase());
      if (factory == null) {
        throw new Error(
            `Unknown value ${type} for flag --backend. ` +
            `Allowed values are ${backendNames}.`);
      }
    }
  });

  if (flags == null && factory == null) {
    return null;
  }
  if (flags != null && factory == null) {
    throw new Error(
        '--backend flag is required when --flags is present. ' +
        `Available values are ${backendNames}.`);
  }
  return {flags: flags || {}, name: backendName, backendName};
}

export function describeWithFlags(
    name: string, constraints: Constraints, tests: (env: TestEnv) => void) {
  TEST_ENVS.forEach(testEnv => {
    ENV.setFlags(testEnv.flags);
    if (envSatisfiesConstraints(
            ENV, testEnv.backendName, ENGINE.backendNames(), constraints)) {
      const testName =
          name + ' ' + testEnv.name + ' ' + JSON.stringify(testEnv.flags);
      executeTests(testName, tests, testEnv);
    }
  });
}

export interface TestEnv {
  name: string;
  backendName: string;
  flags?: Flags;
}

export let TEST_ENVS: TestEnv[] = [];

// Whether a call to setTestEnvs has been called so we turn off
// registration. This allows command line overriding or programmatic
// overriding of the default registrations.
let testEnvSet = false;
export function setTestEnvs(testEnvs: TestEnv[]) {
  testEnvSet = true;
  TEST_ENVS = testEnvs;
}

export function registerTestEnv(testEnv: TestEnv) {
  // When using an explicit call to setTestEnvs, turn off registration of
  // test environments because the explicit call will set the test
  // environments.
  if (testEnvSet) {
    return;
  }
  TEST_ENVS.push(testEnv);
}

if (typeof __karma__ !== 'undefined') {
  const testEnv = parseKarmaFlags(__karma__.config.args);
  if (testEnv != null) {
    setTestEnvs([testEnv]);
  }
}

function executeTests(
    testName: string, tests: (env: TestEnv) => void, testEnv: TestEnv) {
  describe(testName, () => {
    beforeAll(() => {
      ENGINE.reset();
      if (testEnv.flags != null) {
        ENV.setFlags(testEnv.flags);
      }
      ENV.set('IS_TEST', true);
      ENGINE.setBackend(testEnv.backendName);
    });

    beforeEach(() => {
      ENGINE.startScope();
    });

    afterEach(() => {
      ENGINE.endScope();
      ENGINE.disposeVariables();
    });

    afterAll(() => {
      ENGINE.reset();
    });

    tests(testEnv);
  });
}

export class TestKernelBackend extends KernelBackend {
  dispose(): void {}
}
