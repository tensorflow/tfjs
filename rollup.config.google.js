/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import node from 'rollup-plugin-node-resolve';
import commonjs from 'rollup-plugin-commonjs';
import cleanup from 'rollup-plugin-cleanup';

/**
 * This rollup config is used by `yarn gen-google3-proto` to create a
 * self-contained compiled_api.js that has protobuf in it. This script is used
 * only when synching code internally to eliminate having protobufjs as an
 * external dependency.
 */

const PREAMBLE = `/**
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

export default {
  input: 'src/data/compiled_api.js',
    plugins: [
      node(),
      // Polyfill require() from dependencies.
      commonjs({
        namedExports: {
          './node_modules/protobufjs/minimal.js': ['roots', 'Reader', 'util']
        }
      }),
      cleanup({comments: 'none'}),
    ],
    output: {
      banner: PREAMBLE,
      globals: {'@tensorflow/tfjs-core': 'tf'},
      format: 'es',
      file: 'src/data/compiled_api.js'
    },
    external: ['@tensorflow/tfjs-core'],
    onwarn: warning => {
      let {code} = warning;
      if (code === 'CIRCULAR_DEPENDENCY' ||
          code === 'CIRCULAR' || code === 'EVAL') {
        return;
      }
      console.warn('WARNING: ', warning.toString());
    }
};
