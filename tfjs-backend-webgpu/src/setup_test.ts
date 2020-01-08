/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

// tslint:disable-next-line: no-imports-from-dist
import {setTestEnvs} from '@tensorflow/tfjs-core/dist/jasmine_util';

// TODO: Remove and import from tfjs-core once 1.3.2 is released, like so:
// import {setTestEnvs, setupTestFilters, TestFilter} from
// '@tensorflow/tfjs-core/dist/jasmine_util';

interface TestFilter {
  include?: string;
  startsWith?: string;
  excludes?: string[];
}

export function setupTestFilters(
    testFilters: TestFilter[], customInclude: (name: string) => boolean) {
  const env = jasmine.getEnv();
  // Account for --grep flag passed to karma by saving the existing specFilter.
  const grepFilter = env.specFilter;

  /**
   * Filter method that returns boolean, if a given test should run or be
   * ignored based on its name. The exclude list has priority over the
   * include list. Thus, if a test matches both the exclude and the include
   * list, it will be exluded.
   */
  // tslint:disable-next-line: no-any
  env.specFilter = (spec: any) => {
    // Filter out tests if the --grep flag is passed.
    if (!grepFilter(spec)) {
      return false;
    }

    const name = spec.getFullName();

    if (customInclude(name)) {
      return true;
    }

    // Include a describeWithFlags() test from tfjs-core only if the test is
    // in the include list.
    for (let i = 0; i < testFilters.length; ++i) {
      const testFilter = testFilters[i];
      if ((testFilter.include != null &&
           name.indexOf(testFilter.include) > -1) ||
          (testFilter.startsWith != null &&
           name.startsWith(testFilter.startsWith))) {
        if (testFilter.excludes != null) {
          for (let j = 0; j < testFilter.excludes.length; j++) {
            if (name.indexOf(testFilter.excludes[j]) > -1) {
              return false;
            }
          }
        }
        return true;
      }
    }
    // Otherwise ignore the test.
    return false;
  };
}

setTestEnvs([{
  name: 'test-webgpu',
  backendName: 'webgpu',
  flags: {'WEBGPU_CPU_FORWARD': false},
  isDataSync: false,
}]);

const TEST_FILTERS: TestFilter[] = [
  {include: 'deprecated batchNormalization', excludes: []},
];

const customInclude = (testName: string) => {
  // Include regular describe() tests.
  if (testName.indexOf('test-webgpu') < 0) {
    return true;
  }

  // Include webgpu specific tests.
  if (testName.startsWith('webgpu')) {
    return true;
  }

  return false;
};

setupTestFilters(TEST_FILTERS, customInclude);

// Import and run all the tests from core.
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/tests';
