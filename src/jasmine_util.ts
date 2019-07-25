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
  flags?: Flags,
  predicate?: (testEnv: TestEnv) => boolean,
};

export const NODE_ENVS: Constraints = {
  predicate: () => ENV.platformName === 'node'
};
export const CHROME_ENVS: Constraints = {
  flags: {'IS_CHROME': true}
};
export const BROWSER_ENVS: Constraints = {
  predicate: () => ENV.platformName === 'browser'
};

export const SYNC_BACKEND_ENVS: Constraints = {
  predicate: (testEnv: TestEnv) => testEnv.isDataSync === true
};

export const HAS_WORKER = {
  predicate: () => typeof(Worker) !== 'undefined'
      && typeof(Blob) !== 'undefined' && typeof(URL) !== 'undefined'
};

export const HAS_NODE_WORKER = {
  predicate: () => {
    let hasWorker = true;
    try {
      require.resolve('worker_threads');
    } catch {
      hasWorker = false;
    }
    return typeof(process) !== 'undefined' && hasWorker;
  }
};

export const ALL_ENVS: Constraints = {};

// Tests whether the current environment satisfies the set of constraints.
export function envSatisfiesConstraints(
    env: Environment, testEnv: TestEnv, constraints: Constraints): boolean {
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
  if (constraints.predicate != null && !constraints.predicate(testEnv)) {
    return false;
  }
  return true;
}

export function parseTestEnvFromKarmaFlags(
    args: string[], registeredTestEnvs: TestEnv[]): TestEnv {
  let flags: Flags;
  let testEnvName: string;

  args.forEach((arg, i) => {
    if (arg === '--flags') {
      flags = JSON.parse(args[i + 1]);
    } else if (arg === '--testEnv') {
      testEnvName = args[i + 1];
    }
  });

  const testEnvNames = registeredTestEnvs.map(env => env.name).join(', ');
  if (flags != null && testEnvName == null) {
    throw new Error(
        '--testEnv flag is required when --flags is present. ' +
        `Available values are [${testEnvNames}].`);
  }
  if (testEnvName == null) {
    return null;
  }

  let testEnv: TestEnv;
  registeredTestEnvs.forEach(env => {
    if (env.name === testEnvName) {
      testEnv = env;
    }
  });
  if (testEnv == null) {
    throw new Error(
        `Test environment with name ${testEnvName} not ` +
        `found. Available test environment names are ` +
        `${testEnvNames}`);
  }
  if (flags != null) {
    testEnv.flags = flags;
  }

  return testEnv;
}

export function describeWithFlags(
    name: string, constraints: Constraints, tests: (env: TestEnv) => void) {
  if (TEST_ENVS.length === 0) {
    throw new Error(
        `Found no test environments. This is likely due to test environment ` +
        `registries never being imported or test environment registries ` +
        `being registered too late.`);
  }

  TEST_ENVS.forEach(testEnv => {
    ENV.setFlags(testEnv.flags);
    if (envSatisfiesConstraints(ENV, testEnv, constraints)) {
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
  isDataSync?: boolean;
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

function executeTests(
    testName: string, tests: (env: TestEnv) => void, testEnv: TestEnv) {
  describe(testName, () => {
    beforeAll(async () => {
      ENGINE.reset();
      if (testEnv.flags != null) {
        ENV.setFlags(testEnv.flags);
      }
      ENV.set('IS_TEST', true);
      // Await setting the new backend since it can have async init.
      await ENGINE.setBackend(testEnv.backendName);
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
