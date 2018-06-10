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
import babel from 'rollup-plugin-babel';
import node from 'rollup-plugin-node-resolve';
import typescript from 'rollup-plugin-typescript2';
import commonjs from 'rollup-plugin-commonjs';
import resolve from 'rollup-plugin-node-resolve';
import json from 'rollup-plugin-json';
import uglify from 'rollup-plugin-uglify';

const copyright =  `// @tensorflow/tfjs Copyright ${(new Date).getFullYear()} Google`;

function minify() {
  return uglify({
    output: {
      preamble: copyright
    }
  });
}

function config({plugins = [], output = {}, external = []}) {
  return {
    input: 'src/index.ts',
    plugins: [
      typescript({
        tsconfigOverride: {compilerOptions: {module: 'ES2015'}}
      }),
      node(),
      // Polyfill require() from dependencies.
      commonjs({
        ignore: ["crypto"],
        include: 'node_modules/**',
        namedExports: {
          './node_modules/seedrandom/index.js': ['alea'],
          './src/data/compiled_api.js': ['tensorflow'],
          './node_modules/protobufjs/minimal.js': ['roots', 'Reader', 'util']
        },
      }),
      json(),
      // We need babel to compile the compiled_api.js generated proto file from es6 to es5.
      babel(),
      ...plugins
    ],
    output: {
      banner: copyright,
      ...output
    },
    external: [
      'crypto',
      ...external
    ],
    onwarn: warning => {
      let {code} = warning;
      if (code === 'CIRCULAR_DEPENDENCY' ||
          code === 'CIRCULAR' ||
          code === 'THIS_IS_UNDEFINED') {
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
      file: 'dist/tf.js'
    }
  }),
  config({
    plugins: [
      minify()
    ],
    output: {
      format: 'umd',
      name: 'tf',
      extend: true,
      file: 'dist/tf.min.js'
    }
  }),
  config({
    plugins: [
      minify()
    ],
    output: {
      format: 'es',
      file: 'dist/tf.esm.js',
      globals: {
        '@tensorflow/tfjs-core': 'tf',
        '@tensorflow/tfjs-layers': 'tf',
        '@tensorflow/tfjs-converter': 'tf'
      }
    },
    external: [
      '@tensorflow/tfjs-core',
      '@tensorflow/tfjs-layers',
      '@tensorflow/tfjs-converter'
    ]
  })
];
