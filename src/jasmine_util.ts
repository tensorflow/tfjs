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
import {ENV, Environment, Features} from './environment';
import {KernelBackend} from './kernels/backend';
import {MathBackendCPU} from './kernels/backend_cpu';
import {MathBackendWebGL} from './kernels/backend_webgl';
import {NATIVE_ENV} from './test_util';

function canEmulateFeature<K extends keyof Features>(
    feature: K, emulatedFeatures: Features): boolean {
  const emulatedFeature = emulatedFeatures[feature];

  if (feature === 'BACKEND') {
    return ENV.findBackend(emulatedFeature as string) != null;
  } else if (feature === 'WEBGL_VERSION') {
    return ENV.get(feature) >= emulatedFeature;
  } else if (
      feature === 'WEBGL_RENDER_FLOAT32_ENABLED' ||
      feature === 'WEBGL_DOWNLOAD_FLOAT_ENABLED' || feature === 'IS_CHROME') {
    if (ENV.get(feature) === false && emulatedFeature === true) {
      return false;
    }
    return true;
  }
  return true;
}

// Tests whether the set of features can be emulated within the current real
// environment.
export function canEmulateEnvironment(emulatedFeatures: Features): boolean {
  const featureNames = Object.keys(emulatedFeatures) as Array<keyof Features>;
  for (let i = 0; i < featureNames.length; i++) {
    const featureName = featureNames[i];
    if (!canEmulateFeature(featureName, emulatedFeatures)) {
      return false;
    }
  }
  return true;
}

// Checks whether any of the emulated features are equivalent to the default
// environment by comparing the features.
export function anyFeaturesEquivalentToDefault(
    emulatedFeatures: Features[], environment: Environment) {
  for (let j = 0; j < emulatedFeatures.length; j++) {
    const candidateDuplicateFeature = emulatedFeatures[j];
    if (candidateDuplicateFeature === NATIVE_ENV) {
      continue;
    }

    const featureNames =
        Object.keys(candidateDuplicateFeature) as Array<(keyof Features)>;
    const featuresMatch = featureNames.every(
        featureName => candidateDuplicateFeature[featureName] ===
            environment.get(featureName));

    if (featuresMatch) {
      return true;
    }
  }
  return false;
}

export function describeWithFlags(
    name: string, featuresToRun: Features[], tests: () => void) {
  registerTestBackends();

  for (let i = 0; i < featuresToRun.length; i++) {
    const features = featuresToRun[i];
    // If using the default feature, check for duplicates and don't execute the
    // default if it's a duplicate.
    if (features === NATIVE_ENV &&
        anyFeaturesEquivalentToDefault(featuresToRun, ENV)) {
      continue;
    }

    if (canEmulateEnvironment(features)) {
      const testName = name + ' ' + JSON.stringify(features);
      executeTests(testName, tests, features);
    }
  }
}

export interface TestBackendFactory {
  name: string;
  factory: () => KernelBackend;
  priority: number;
}

export let TEST_BACKENDS: TestBackendFactory[];
setTestBackends([
  // High priority to override the real defaults.
  {name: 'test-webgl', factory: () => new MathBackendWebGL(), priority: 101},
  {name: 'test-cpu', factory: () => new MathBackendCPU(), priority: 100}
]);

let BEFORE_ALL = (features: Features) => {};
let AFTER_ALL = (features: Features) => {};
let BEFORE_EACH = (features: Features) => {};
let AFTER_EACH = (features: Features) => {};
export function setBeforeAll(f: (features: Features) => void) {
  BEFORE_ALL = f;
}
export function setAfterAll(f: (features: Features) => void) {
  AFTER_ALL = f;
}
export function setBeforeEach(f: (features: Features) => void) {
  BEFORE_EACH = f;
}
export function setAfterEach(f: (features: Features) => void) {
  AFTER_EACH = f;
}
export function setTestBackends(testBackends: TestBackendFactory[]) {
  TEST_BACKENDS = testBackends;
}
export function registerTestBackends() {
  TEST_BACKENDS.forEach(testBackend => {
    if (ENV.findBackend(testBackend.name) != null) {
      ENV.removeBackend(testBackend.name);
    }
    ENV.registerBackend(
        testBackend.name, testBackend.factory, testBackend.priority);
  });
}

function executeTests(testName: string, tests: () => void, features: Features) {
  describe(testName, () => {
    beforeAll(() => {
      ENV.setFeatures(features);
      registerTestBackends();

      BEFORE_ALL(features);
    });

    beforeEach(() => {
      BEFORE_EACH(features);
      if (features && features.BACKEND != null) {
        Environment.setBackend(features.BACKEND);
      }
      ENV.engine.startScope();
    });

    afterEach(() => {
      ENV.engine.endScope(null);
      AFTER_EACH(features);
    });

    afterAll(() => {
      AFTER_ALL(features);

      ENV.reset();
    });

    tests();
  });
}
