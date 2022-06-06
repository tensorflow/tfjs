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

import commonjs from '@rollup/plugin-commonjs';
import resolve from '@rollup/plugin-node-resolve';
import sourcemaps from 'rollup-plugin-sourcemaps';
import visualizer from 'rollup-plugin-visualizer';

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
  globals = [],
  external = [],
  leave_as_require = [],
  plugins = [],
  vis_filename,
}: {
  globals?: string[],
  external?: string[],
  leave_as_require?: string[],
  plugins?: unknown[],
  vis_filename: string,
}) {
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
      visualizer({
        sourcemap: true,
        filename: vis_filename,
        template: 'sunburst',
      }),
    ],
    onwarn: function (warning: {code: string, message: string}) {
      if (warning.code === 'THIS_IS_UNDEFINED') {
        return;
      }
      console.warn(warning.message);
    }
  }
}
