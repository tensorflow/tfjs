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
import {terser} from 'rollup-plugin-terser';
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
        namedExports: {
          './node_modules/protobufjs/minimal.js':
              ['roots', 'Reader', 'util']
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
    external: ['@tensorflow/tfjs-core', ...external],
    onwarn: warning => {
      let {code} = warning;
      if (code === 'CIRCULAR_DEPENDENCY' || code === 'CIRCULAR' ||
          code === 'EVAL') {
        return;
      }
      console.warn('WARNING: ', warning.toString());
    }
  };
}

module.exports = cmdOptions => {
  const bundles = [];

  const terserPlugin = terser({output: {preamble: PREAMBLE, comments: false}});
  const name = 'tf';
  const extend = true;
  const browserFormat = 'umd';
  const fileName = 'tf-converter';

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

  if (cmdOptions.ci || cmdOptions.npm) {
    // Browser default minified (ES5)
    bundles.push(config({
      plugins: [terserPlugin],
      output: {
        format: browserFormat,
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
    // Browser default unminified (ES5)
    bundles.push(config({
      output: {
        format: browserFormat,
        name,
        extend,
        file: `dist/${fileName}.js`,
        freeze: false
      },
      tsCompilerOptions: {target: 'es5'}
    }));

    // Browser ES2017
    bundles.push(config({
      output: {
        format: browserFormat,
        name,
        extend,
        file: `dist/${fileName}.es2017.js`
      },
      tsCompilerOptions: {target: 'es2017'}
    }));

    // Browser ES2017 minified
    bundles.push(config({
      plugins: [terserPlugin],
      output: {
        format: browserFormat,
        name,
        extend,
        file: `dist/${fileName}.es2017.min.js`
      },
      tsCompilerOptions: {target: 'es2017'}
    }));
  }

  return bundles;
};
