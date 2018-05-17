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
import resolve from 'rollup-plugin-node-resolve';

export default {
  input: 'src/index.ts',
  plugins: [
    typescript(),
    node(),
    // Polyfill require() from dependencies.
    commonjs({
      include: 'node_modules/**',
      namedExports: {
        './node_modules/seedrandom/index.js': ['alea']
      },
    })
  ],
  output: {
    extend: true,
    banner: `// @tensorflow/tfjs-layers Copyright ${(new Date).getFullYear()} Google`,
    file: 'dist/tf-layers.js',
    format: 'umd',
    name: 'tf',
    globals: {'crypto': 'crypto', '@tensorflow/tfjs-core': 'tf'}
  },
  external: ['crypto', '@tensorflow/tfjs-core'],
  onwarn: warning => {
    let {code} = warning;
    if (code === 'CIRCULAR_DEPENDENCY' ||
        code === 'CIRCULAR') {
      return;
    }
    console.warn('WARNING: ', warning.toString());
  }
};
