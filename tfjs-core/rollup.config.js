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
import {terser} from 'rollup-plugin-terser';
import typescript from 'rollup-plugin-typescript2';
import visualizer from 'rollup-plugin-visualizer';

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

function config({plugins = [], output = {}, external = [], visualize = false}) {
  if (visualize) {
    const filename = output.file + '.html';
    plugins.push(visualizer({
      sourcemap: true,
      filename,
    }));
    console.log(`Will output a bundle visualization in ${filename}`);
  }
  return {
    input: 'src/index.ts',
    plugins: [
      typescript({
        tsconfigOverride: {compilerOptions: {module: 'ES2015'}},
        // See https://github.com/ezolenko/rollup-plugin-typescript2/issues/105
        objectHashIgnoreUnknownHack: visualize ? true : false,
        clean: visualize ? true : false,
      }),
      node(),
      // Polyfill require() from dependencies.
      commonjs({
        ignore: ['crypto', 'node-fetch', 'util'],
        include: 'node_modules/**',
        namedExports: {
          './node_modules/seedrandom/index.js': ['alea'],
        },
      }),
      ...plugins
    ],
    output: {
      banner: PREAMBLE,
      sourcemap: true,
      ...output,
    },
    external: ['crypto', ...external],
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

module.exports = cmdOptions => {
  const bundles = [];

  if (!cmdOptions.ci) {
    // tf-core.js
    bundles.push(config({
      output: {
        format: 'umd',
        name: 'tf',
        extend: true,
        file: 'dist/tf-core.js',
      }
    }));
  }

  // tf-core.min.js
  bundles.push(config({
    plugins: [terser({output: {preamble: PREAMBLE}})],
    output: {
      format: 'umd',
      name: 'tf',
      extend: true,
      file: 'dist/tf-core.min.js',
    },
    visualize: cmdOptions.visualize
  }));

  if (!cmdOptions.ci) {
    // tf-core.esm.js
    bundles.push(config({
      plugins: [terser({output: {preamble: PREAMBLE}})],
      output: {
        format: 'es',
        file: 'dist/tf-core.esm.js',
      }
    }));
  }
  return bundles;
};
