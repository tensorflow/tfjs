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
import typescript from 'rollup-plugin-typescript2';
import commonjs from 'rollup-plugin-commonjs';
import uglify from 'rollup-plugin-uglify';

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

function minify() {
  return uglify({
    output: {preamble: PREAMBLE}
  });
}

function config({plugins = [], output = {}}) {
  return {
    input: 'src/index.ts',
    plugins: [
      typescript({
        tsconfigOverride: {compilerOptions: {module: 'ES2015'}}
      }),
      node(),
      // Polyfill require() from dependencies.
      commonjs({
        namedExports: {
          './src/data/compiled_api.js': ['tensorflow'],
          './node_modules/protobufjs/minimal.js': ['roots', 'Reader', 'util'],
          './node_modules/base64/index.js': ['Base64']
        }
      }),
      ...plugins
    ],
    output: {
      banner: PREAMBLE,
      sourcemap: true,
      globals: {'@tensorflow/tfjs-core': 'tf'},
      ...output
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
}

export default [
  config({
    output: {
      format: 'umd',
      name: 'tf',
      extend: true,
      file: 'dist/tf-converter.js'
    }
  }),
  config({
    plugins: [minify()],
    output: {
      format: 'umd',
      name: 'tf',
      extend: true,
      file: 'dist/tf-converter.min.js'
    }
  }),
  config({
    plugins: [minify()],
    output: {
      format: 'es',
      file: 'dist/tf-converter.esm.js'
    }
  })
];
