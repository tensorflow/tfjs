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

import commonjs from 'rollup-plugin-commonjs';
import node from 'rollup-plugin-node-resolve';
import {terser} from 'rollup-plugin-terser';
import typescript from 'rollup-plugin-typescript2';

const PREAMBLE = ``;

function config({plugins = [], output = {}, external = []}) {
  return {
    input: 'src/index.ts',
    plugins: [
      typescript({tsconfigOverride: {compilerOptions: {module: 'ES2015'}}}),
      node(),
      // Polyfill require() from dependencies.
      commonjs({
        ignore: ['crypto'],
        include: 'node_modules/**',
      }),
      ...plugins
    ],
    output: {
      banner: PREAMBLE,
      sourcemap: true,
      globals: {'@tensorflow/tfjs-core': 'tf'},
      ...output,
    },
    external: [
      '@tensorflow/tfjs-core',
      '@react-native-community/async-storage',
      'react-native',
    ],
    onwarn: warning => {
      let {code} = warning;
      if (code === 'CIRCULAR_DEPENDENCY' || code === 'CIRCULAR' ||
          code === 'THIS_IS_UNDEFINED') {
        return;
      }
      console.warn('WARNING: ', warning.toString());
    }
  };
}

module.exports = cmdOptions => [
    // unminified umd
    config({
      output: {
        format: 'umd',
        name: 'tf',
        extend: true,
        file: 'dist/tf-react-native.js',
      }
    }),

    // minified umd
    config({
      plugins: [terser({output: {preamble: PREAMBLE}})],
      output: {
        format: 'umd',
        name: 'tf',
        extend: true,
        file: 'dist/tf-react-native.min.js',
      },
      visualize: cmdOptions.visualize
    }),

    // esm bundler
    config({
      plugins: [terser({output: {preamble: PREAMBLE}})],
      output: {
        format: 'es',
        file: 'dist/tf-react-native.esm.js',
      }
    }),
];
