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
import typescript from '@rollup/plugin-typescript';
import {terser} from 'rollup-plugin-terser';
import visualizer from 'rollup-plugin-visualizer';

const PREAMBLE = ``;

function config({
  plugins = [],
  output = {},
  external = [],
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
      typescript(tsoptions), resolve(),
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
    external: ['@tensorflow/tfjs-core'],
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
  const name = 'tf';
  const extend = true;
  const browserFormat = 'umd';
  const fileName = 'tf-webgpu';

  // Browser default unminified
  bundles.push(config({
    output: {
      format: browserFormat,
      name,
      extend,
      file: `dist/${fileName}.js`,
    }
  }));

  // Browser default minified
  bundles.push(config({
    plugins: [terser({output: {preamble: PREAMBLE}})],
    output: {
      format: browserFormat,
      name,
      extend,
      file: `dist/${fileName}.min.js`,
    },
    visualize: cmdOptions.visualize
  }));

  // Browser ES2017
  bundles.push(config({
    output: {
      format: browserFormat,
      name,
      extend,
      file: `dist/${fileName}.es2017.js`,
    },
    tsCompilerOptions: {target: 'es2017'}
  }));

  // Browser ES2017 minified
  bundles.push(config({
    plugins: [terser({output: {preamble: PREAMBLE}})],
    output: {
      format: browserFormat,
      name,
      extend,
      file: `dist/${fileName}.es2017.min.js`,
    },
    tsCompilerOptions: {target: 'es2017'}
  }));

  return bundles;
};
