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

import * as commonjs_import from '@rollup/plugin-commonjs';
import resolve from '@rollup/plugin-node-resolve';
import * as sourcemaps_import from 'rollup-plugin-sourcemaps';
import * as visualizer from 'rollup-plugin-visualizer';
import {terser as terserPlugin} from 'rollup-plugin-terser';
import {downlevelToEs5Plugin} from 'downlevel_to_es5_plugin/downlevel_to_es5_plugin';

// These workarounds would not be necessary with esModuleInterop = true,
// but we likely can't set that.
const commonjs = commonjs_import as unknown as typeof commonjs_import.default;
const sourcemaps =
  sourcemaps_import as unknown as typeof sourcemaps_import.default;

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

export function makeRollupConfig({
  globals = {},
  external = [],
  leave_as_require = [],
  plugins = [],
  terser = false,
  es5 = false,
  vis_filename,
}: {
  globals?: {[name: string]: string},
  external?: string[],
  leave_as_require?: string[],
  plugins?: unknown[],
  terser?: boolean,
  es5?: boolean,
  vis_filename?: string,
}) {
  const vis = vis_filename ? [visualizer({
    sourcemap: true,
    filename: vis_filename,
    template: 'sunburst',
  })] : [];

  // Without `compress: {typeofs: false}`, the terser plugin will turn
  // `typeof _scriptDir == "undefined"` into `_scriptDir === void 0` in minified
  // JS file which will cause "_scriptDir is undefined" error in web worker's
  // inline script.
  //
  // For more context, see tfjs-backend-wasm/scripts/patch-threaded-simd-module.js
  const useTerser = terser ? [
    terserPlugin({
      output: {preamble, comments: false},
      compress: {typeofs: false},
    })
  ] : [];

  const useEs5 = es5 ? [downlevelToEs5Plugin] : [];

  return {
    output: {
      banner: preamble,
      freeze: false, // For tests that spyOn imports
      extend: true, // For imports that extend the global 'tf' variable
      globals,
    },
    external,
    plugins: [
      resolve({browser: true}),
      commonjs({
        ignore: leave_as_require,
      }),
      sourcemaps(),
      ...plugins,
      ...useEs5,
      ...useTerser,
      ...vis,
    ],
    onwarn: function (warning: {code: string, message: string}) {
      if (warning.code === 'THIS_IS_UNDEFINED') {
        return;
      }
      console.warn(warning.message);
    }
  }
}
