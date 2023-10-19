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

import commonjs from '@rollup/plugin-commonjs';
import resolve from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';
import babel from '@rollup/plugin-babel';
import {terser} from 'rollup-plugin-terser';
import {visualizer} from 'rollup-plugin-visualizer';

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

function config({
  plugins = [],
  output = {},
  external = [],
  visualize = false,
  tsCompilerOptions = {},
  entry = 'src/index.ts'
}) {
  if (visualize) {
    const filename = output.file + '.html';
    plugins.push(visualizer({sourcemap: true, filename}));
    console.log(`Will output a bundle visualization in ${filename}`);
  }

  const defaultTsOptions = {
    include: ['src/**/*.ts'],
    module: 'ES2015',
  };
  const tsoptions = Object.assign({}, defaultTsOptions, tsCompilerOptions);

  return {
    input: entry,
    // Do not tree shake union package library
    treeshake: false,
    plugins: [
      typescript(tsoptions),
      resolve({dedupe: ['seedrandom', 'long']}),
      commonjs({
        ignore: ['crypto', 'fs', 'node-fetch', 'string_decoder', 'util'],
        include: 'node_modules/**',
      }),
      ...plugins,
    ],
    output: {
      banner: PREAMBLE,
      globals: {
        'node-fetch': 'nodeFetch',
      },
      sourcemap: true,
      ...output
    },
    external: [
      // node-fetch is only used in node. Browsers have native "fetch".
      'node-fetch', 'crypto', ...external
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

export default function(cmdOptions) {
  const bundles = [];

  const babelPlugin = babel({
    babelrc: false,
    babelHelpers: 'bundled',
    presets: [
      // ensure we get es5 by adding IE 11 as a target
      [
        '@babel/env', {
          modules: false,
          useBuiltIns: 'entry',
          corejs: '3.29.1',
          targets: {'ie': '11'},
        }
      ]
    ]
  });

  const terserPlugin = terser({output: {preamble: PREAMBLE, comments: false}});
  const name = 'tf';
  const extend = true;
  const umdFormat = 'umd';
  const fesmFormat = 'es';
  const fileName = 'tf';

  // Node
  bundles.push(config({
    output: {
      format: 'cjs',
      name,
      extend,
      file: `dist/${fileName}.node.js`,
      freeze: false
    },
    tsCompilerOptions: {target: 'es5'},
    external: [
      'long',
      '@tensorflow/tfjs-core',
      '@tensorflow/tfjs-layers',
      '@tensorflow/tfjs-converter',
      '@tensorflow/tfjs-data',
      '@tensorflow/tfjs-backend-cpu',
      '@tensorflow/tfjs-backend-webgl',
    ]
  }));

  if (cmdOptions.ci || cmdOptions.npm) {
    // UMD ES5
    bundles.push(config({
      entry: 'src/index_with_polyfills.ts',
      plugins: [babelPlugin, terserPlugin],
      output: {
        format: umdFormat,
        name,
        extend,
        file: `dist/${fileName}.min.js`,
        freeze: false
      },
      tsCompilerOptions: {target: 'es5'},
      visualize: cmdOptions.visualize
    }));
  }

  if (cmdOptions.npm) {
    // UMD ES5
    bundles.push(config({
      entry: 'src/index_with_polyfills.ts',
      plugins: [babelPlugin],
      output: {
        format: umdFormat,
        name,
        extend,
        file: `dist/${fileName}.js`,
        freeze: false
      },
      tsCompilerOptions: {target: 'es5'}
    }));

    // UMD ES2017
    bundles.push(config({
      output:
          {format: umdFormat, name, extend, file: `dist/${fileName}.es2017.js`},
      tsCompilerOptions: {target: 'es2017'}
    }));

    // UMD ES2017 minified
    bundles.push(config({
      plugins: [terserPlugin],
      output: {
        format: umdFormat,
        name,
        extend,
        file: `dist/${fileName}.es2017.min.js`
      },
      tsCompilerOptions: {target: 'es2017'},
      visualize: cmdOptions.visualize
    }));

    // FESM ES2017
    bundles.push(config({
      output:
          {format: fesmFormat, name, extend, file: `dist/${fileName}.fesm.js`},
      tsCompilerOptions: {target: 'es2017'}
    }));

    // FESM ES2017 minified
    bundles.push(config({
      plugins: [terserPlugin],
      output: {
        format: fesmFormat,
        name,
        extend,
        file: `dist/${fileName}.fesm.min.js`
      },
      tsCompilerOptions: {target: 'es2017'},
      visualize: cmdOptions.visualize
    }));
  }

  return bundles;
};
