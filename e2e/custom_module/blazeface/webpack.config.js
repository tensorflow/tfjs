/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

const path = require('path');
const TerserPlugin = require('terser-webpack-plugin');

module.exports = function(env) {
  const outputPath = (env && env.useCustomTfjs) ? 'dist/custom' : 'dist/full'

  const config = {
    mode: 'production',
    entry: './app.js',
    target: 'web',
    output: {
      path: path.resolve(__dirname, outputPath),
      filename: 'app_webpack.js',
    },
    optimization: {
      minimizer: [
        new TerserPlugin({
          cache: true,
          parallel: true,
          sourceMap: false,
          terserOptions: {
            comments: false,
          }
        }),
      ]
    },
    module: {
      rules: [
        {
          test: /\.wasm$/i,
          type: 'javascript/auto',
          use: [
            {
              loader: 'file-loader',
            },
          ],
        },
      ],
    }
  };

  if (env && env.useCustomTfjs) {
    config.resolve = {
      alias: {
        '@tensorflow/tfjs$':
            path.resolve(__dirname, './custom_tfjs_blazeface/custom_tfjs.js'),
        '@tensorflow/tfjs-core$': path.resolve(
            __dirname, './custom_tfjs_blazeface/custom_tfjs_core.js'),
        '@tensorflow/tfjs-core/dist/ops/ops_for_converter': path.resolve(
            __dirname, './custom_tfjs_blazeface/custom_ops_for_converter.js'),
      }
    }
  }
  return config;
}
