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

// Register Layers' flags.
import './flags_layers';
import '@tensorflow/tfjs-core';
// tslint:disable-next-line:no-imports-from-dist
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/register_all_gradients';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

// tslint:disable-next-line: no-imports-from-dist
import {parseTestEnvFromKarmaFlags, registerTestEnv, setTestEnvs, TEST_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

// Register test environments.
const webgl1TestEnv = {
  name: 'webgl1',
  backendName: 'webgl',
  flags: {
    'WEBGL_VERSION': 1,
    'WEBGL_CPU_FORWARD': false,
    'WEBGL_SIZE_UPLOAD_UNIFORM': 0
  },
  isDataSync: true
};
registerTestEnv(webgl1TestEnv);
registerTestEnv({name: 'cpu', backendName: 'cpu'});
registerTestEnv({
  name: 'webgl2',
  backendName: 'webgl',
  flags: {
    'WEBGL_VERSION': 2,
    'WEBGL_CPU_FORWARD': false,
    'WEBGL_SIZE_UPLOAD_UNIFORM': 0
  }
});

// Allow flags to override test envs
// tslint:disable-next-line:no-any
declare let __karma__: any;
if (typeof __karma__ !== 'undefined') {
  const testEnv = parseTestEnvFromKarmaFlags(__karma__.config.args, TEST_ENVS);

  if (testEnv != null) {
    setTestEnvs([testEnv]);
  } else {
    // Exclude webgl1 unless it is specifically requested because it causes
    // test flakiness when switching between webgl1 and webgl2.
    setTestEnvs(TEST_ENVS.filter(env => env !== webgl1TestEnv));
  }
}

// Import and run tests from layers.
// tslint:disable-next-line:no-require-imports
require('./tests');
