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
import '@tensorflow/tfjs-backend-webgl';

// tslint:disable-next-line: no-imports-from-dist
import {parseTestEnvFromKarmaFlags, registerTestEnv, setTestEnvs, TEST_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';

// Increase test timeout since we are fetching the model files from GCS.
jasmine.DEFAULT_TIMEOUT_INTERVAL = 20000;

registerTestEnv({
  name: 'webgl1',
  backendName: 'webgl',
  flags: {
    'WEBGL_VERSION': 1,
    'WEBGL_CPU_FORWARD': false,
    'WEBGL_SIZE_UPLOAD_UNIFORM': 0
  },
  isDataSync: true
});

// tslint:disable-next-line:no-any
declare let __karma__: any;
const testEnv = parseTestEnvFromKarmaFlags(__karma__.config.args, TEST_ENVS);
if (testEnv != null) {
  setTestEnvs([testEnv]);
} else {
  // Run browser tests againts both the webgl backends.
  setTestEnvs([
    // WebGL.
    {
      name: 'test-webgl',
      backendName: 'webgl',
      flags: {
        'WEBGL_VERSION': 2,
        'WEBGL_CPU_FORWARD': false,
        'WEBGL_SIZE_UPLOAD_UNIFORM': 0
      },
      isDataSync: true
    }
  ]);
}
