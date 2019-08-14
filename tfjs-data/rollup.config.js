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

import commonjs from 'rollup-plugin-commonjs';
import node from 'rollup-plugin-node-resolve';
import typescript from 'rollup-plugin-typescript2';
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
    output: {
      preamble: PREAMBLE
    }
  });
}

function config({
  plugins = [],
  output = {},
  external = []
}) {
  return {
    input: 'src/index.ts',
    plugins: [
      typescript({
        tsconfigOverride: {
          compilerOptions: {
            module: 'ES2015'
          }
        }
      }),
      node(),
      // Polyfill require() from dependencies.
      commonjs({
        ignore: ['crypto', 'node-fetch'],
        include: 'node_modules/**',
        namedExports: {
          './node_modules/seedrandom/index.js': ['alea'],
        },
      }),
      ...plugins
    ],
    output: {
      banner: PREAMBLE,
      globals: {
        'node-fetch': 'nodeFetch',
        '@tensorflow/tfjs-core': 'tf',
      },
      sourcemap: true,
      ...output
    },
    external: [
      // node-fetch is only used in node. Browsers have native "fetch".
      'node-fetch',
      'crypto',
      '@tensorflow/tfjs-core',
      ...external,
    ],
    onwarn: warning => {
      console.warn('WARNING: ', warning.toString());
    }
  };
}

export default [
  config({
    output: {
      format: 'umd',
      name: 'tf.data',
      extend: true,
      file: 'dist/tf-data.js'
    }
  }),
  config({
    plugins: [minify()],
    output: {
      format: 'umd',
      name: 'tf.data',
      extend: true,
      file: 'dist/tf-data.min.js'
    }
  }),
  config({
    plugins: [minify()],
    output: {
      format: 'es',
      file: 'dist/tf-data.esm.js'
    }
  })
];
