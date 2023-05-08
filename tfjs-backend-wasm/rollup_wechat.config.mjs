/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {patchWechatWebAssembly} from './scripts/patch_wechat_webassembly.js';
import {makeRollupConfig} from 'make_rollup_config/make_rollup_config.js';

export default makeRollupConfig({
  globals: {
    '@tensorflow/tfjs-core': 'tf',
    'fs': 'fs',
    'path': 'path',
    'perf_hooks': 'perf_hooks',
    'worker_threads': 'worker_threads',
  },
  external: [
    'crypto',
    '@tensorflow/tfjs-core',
    'fs',
    'path',
    'worker_threads',
    'perf_hooks',
  ],
  leave_as_require: [
    'crypto',
    'node-fetch',
    'util',
    'fs',
    'path',
    'worker_threads',
    'perf_hooks',
    'os',
  ],
  es5: true,
  plugins: [patchWechatWebAssembly()],
});
