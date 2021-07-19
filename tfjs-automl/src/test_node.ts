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
import '@tensorflow/tfjs-backend-cpu';
// tslint:disable-next-line: no-imports-from-dist
import {setTestEnvs} from '@tensorflow/tfjs-core/dist/jasmine_util';

// tslint:disable-next-line:no-require-imports
const jasmine = require('jasmine');

// Increase test timeout since we are fetching the model files from GCS.
jasmine.DEFAULT_TIMEOUT_INTERVAL = 20000;

process.on('unhandledRejection', e => {
  throw e;
});

// Run node tests againts the cpu backend.
setTestEnvs([{name: 'node', backendName: 'cpu'}]);

const runner = new jasmine();
runner.loadConfig({spec_files: ['src/**/*_test.ts'], random: false});
runner.execute();
