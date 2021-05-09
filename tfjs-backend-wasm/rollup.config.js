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

import commonjs from '@rollup/plugin-commonjs';
import resolve from '@rollup/plugin-node-resolve';
import node from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';
import visualizer from 'rollup-plugin-visualizer';
import {getBrowserBundleConfigOptions} from '../rollup.config.helpers';
import {patchWechatWebAssembly} from './scripts/patch-wechat-webassembly'

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
  ignore = [],
  visualize = false,
  tsCompilerOptions = {}
}) {
  if (visualize) {
    const filename = output.file + '.html';
    plugins.push(visualizer(
        {sourcemap: true, filename, template: 'sunburst', gzipSize: true}));
    console.log(`Will output a bundle visualization in ${filename}`);
  }

  const defaultTsOptions = {
    include: ['src/**/*.ts'],
    module: 'ES2015',
  };
  const tsoptions = Object.assign({}, defaultTsOptions, tsCompilerOptions);

  return {
    input: 'src/index.ts',
    plugins: [
      typescript(tsoptions), resolve(), node({preferBuiltins: false}),
      // Polyfill require() from dependencies.
      commonjs({
        ignore: ['crypto', 'node-fetch', 'util', ...ignore],
        include: ['node_modules/**', 'wasm-out/**']
      }),
      ...plugins
    ],
    output: {
      banner: PREAMBLE,
      sourcemap: true,
      globals: {
        '@tensorflow/tfjs-core': 'tf',
        'fs': 'fs',
        'path': 'path',
        'worker_threads': 'worker_threads',
        'perf_hooks': 'perf_hooks'
      },
      ...output,
    },
    external: [
      'crypto',
      '@tensorflow/tfjs-core',
      'fs',
      'path',
      'worker_threads',
      'perf_hooks',
      ...external,
    ],
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

  // Without this, the terser plugin will turn `typeof _scriptDir ==
  // "undefined"` into `_scriptDir === void 0` in minified JS file which will
  // cause "_scriptDir is undefined" error in web worker's inline script.
  //
  // For more context, see scripts/patch-threaded-simd-module.js.
  const terserExtraOptions = {compress: {typeofs: false}};
  if (cmdOptions.npm) {
    const browserBundles = getBrowserBundleConfigOptions(
        config, name, fileName, PREAMBLE, cmdOptions.visualize, false /* CI */,
        terserExtraOptions);
    bundles.push(...browserBundles);
    // Wechat miniprogram
    bundles.push(config({
      output: {
        format: 'cjs',
        name,
        extend,
        file: `dist/miniprogram/index.js`,
        freeze: false
      },
      ignore: ['fs', 'path', 'worker_threads', 'perf_hooks', 'os'],
      tsCompilerOptions: {target: 'es6'},
      plugins: [ patchWechatWebAssembly() ]
    }));
  } else {
    const browserBundles = getBrowserBundleConfigOptions(
        config, name, fileName, PREAMBLE, cmdOptions.visualize, true /* CI */,
        terserExtraOptions);
    bundles.push(...browserBundles);
  }

  return bundles;
};
