/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import '@tensorflow/tfjs-backend-cpu';
// Register the backend.
import './index';
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/register_all_gradients';
import {registerTestEnvs} from './backend_webgl_test_registry';
// tslint:disable-next-line: no-imports-from-dist
import {parseTestEnvFromKarmaFlags, setTestEnvs, setupTestFilters, TEST_ENVS, TestFilter} from '@tensorflow/tfjs-core/dist/jasmine_util';

registerTestEnvs();

const TEST_FILTERS: TestFilter[] = [];
const customInclude = (testName: string) => {
  const toExclude = [
    'isBrowser: false', 'dilation gradient',
    'throws when index is out of bound',
    // otsu tests for threshold op is failing on windows
    'method otsu'
  ];
  for (const subStr of toExclude) {
    if (testName.includes(subStr)) {
      return false;
    }
  }
  // TODO(msoulanille): Prefer TEST_FILTERS over customInclude.
  return true;
};
setupTestFilters(TEST_FILTERS, customInclude);

// Allow flags to override test envs
// tslint:disable-next-line:no-any
declare let __karma__: any;
if (typeof __karma__ !== 'undefined') {
  const testEnv = parseTestEnvFromKarmaFlags(__karma__.config.args, TEST_ENVS);
  if (testEnv != null) {
    setTestEnvs([testEnv]);
  }
}

// These use 'require' because they must not be hoisted above
// the preceding snippet that parses test environments.
// Import and run tests from core.
// tslint:disable-next-line:no-imports-from-dist
// tslint:disable-next-line:no-require-imports
require('@tensorflow/tfjs-core/dist/tests');
// Import and run tests from webgl.
// tslint:disable-next-line:no-imports-from-dist
// tslint:disable-next-line:no-require-imports
require('./tests');
