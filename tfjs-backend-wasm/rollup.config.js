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

function config({plugins = [], output = {}, tsCompilerOptions = {}}) {
  const defaultTsCompilerOptions = {
    module: 'ES2015',
  };
  const tsOverrideCompilerOptions =
      Object.assign({}, defaultTsCompilerOptions, tsCompilerOptions);

  return {
    input: 'src/index.ts',
    plugins: [
      typescript({
        tsconfigOverride: {compilerOptions: tsOverrideCompilerOptions},
      }),
      node({preferBuiltins: true}),
      // Polyfill require() from dependencies.
      commonjs({
        ignore: ['crypto', 'node-fetch', 'util'],
        include: ['node_modules/**', 'wasm-out/**']
      }),
      ...plugins
    ],
    output: {
      banner: PREAMBLE,
      sourcemap: true,
      globals: {'@tensorflow/tfjs-core': 'tf', 'fs': 'fs', 'path': 'path'},
      ...output,
    },
    external: ['crypto', '@tensorflow/tfjs-core', 'fs', 'path'],
    onwarn: warning => {
      let {code} = warning;
      if (code === 'CIRCULAR_DEPENDENCY' || code === 'CIRCULAR' ||
          code === 'THIS_IS_UNDEFINED') {
        return;
      }
      console.warn('WARNING: ', warning.message);
    }
  };
}

module.exports = cmdOptions => {
  const bundles = [];

  const terserPlugin = terser({output: {preamble: PREAMBLE, comments: false}});
  const name = 'tf.wasm';
  const extend = true;
  const fileName = 'tf-backend-wasm';

  // Node
  bundles.push(config({
    output: {
      format: 'cjs',
      name,
      extend,
      file: `dist/${fileName}.node.js`,
      freeze: false
    },
    tsCompilerOptions: {target: 'es5'}
  }));

  if (!cmdOptions.ci) {
    // tf-backend-wasm.js
    bundles.push(config({
      output: {
        format: 'umd',
        name,
        extend,
        file: `dist/${fileName}.js`,
      }
    }));
  }

  // tf-backend-wasm.min.js
  bundles.push(config({
    plugins: [terserPlugin],
    output: {
      format: 'umd',
      name,
      extend,
      file: `dist/${fileName}.min.js`,
    },
  }));

  if (!cmdOptions.ci) {
    // tf-backend-wasm.esm.js
    bundles.push(config({
      plugins: [terserPlugin],
      output: {
        format: 'es',
        file: `dist/${fileName}.esm.js`,
      }
    }));
  }
  return bundles;
};
