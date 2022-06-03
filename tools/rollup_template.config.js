/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import commonjs from '@rollup/plugin-commonjs';
import resolve from '@rollup/plugin-node-resolve';
import sourcemaps from 'rollup-plugin-sourcemaps';
import {terser} from 'rollup-plugin-terser';
import visualizer from 'rollup-plugin-visualizer';
import * as ts from 'typescript';
import path from 'path';
import {downlevelToEs5Plugin} from 'downlevel_to_es5_plugin/downlevel_to_es5_plugin';

const preamble = `/**
 * @license
 * Copyright ${(new Date).getFullYear()} Google LLC. All Rights Reserved.
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
 */`;

const useEs5 = TEMPLATE_es5 ? [downlevelToEs5Plugin] : [];

// Without `compress: {typeofs: false}`, the terser plugin will turn
// `typeof _scriptDir == "undefined"` into `_scriptDir === void 0` in minified
// JS file which will cause "_scriptDir is undefined" error in web worker's
// inline script.
//
// For more context, see tfjs-backend-wasm/scripts/patch-threaded-simd-module.js
const useTerser = TEMPLATE_minify ? [
  terser({
    output: {preamble, comments: false},
    compress: {typeofs: false},
  })
] : [];

export default {
  output: {
    banner: preamble,
    freeze: false, // For tests that spyOn imports
    extend: true, // For imports that extend the global 'tf' variable
    globals: TEMPLATE_globals,
  },
  external: TEMPLATE_external,
  plugins: [
    resolve({browser: true}),
    commonjs({
      ignore: TEMPLATE_leave_as_require,
    }),
    sourcemaps(),
    ...useEs5,
    ...useTerser,
    visualizer({
      sourcemap: true,
      filename: 'TEMPLATE_stats',
      template: 'sunburst',
    }),
  ],
  onwarn: function (warning) {
    if (warning.code === 'THIS_IS_UNDEFINED') {
      return;
    }
    console.warn(warning.message);
  }
}
