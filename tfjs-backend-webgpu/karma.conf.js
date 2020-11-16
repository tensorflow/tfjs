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

const karmaTypescriptConfig = {
  tsconfig: 'tsconfig.test.json',
  // Disable coverage reports and instrumentation by default for tests
  coverageOptions: {instrumentation: false},
  reports: {},
  bundlerOptions: {
    transforms: [require('karma-typescript-es6-transform')({
      presets: [
        // ensure we get es5 by adding IE 11 as a target
        ['@babel/env', {'targets': {'ie': '11'}, 'loose': true}]
      ]
    })],
    // worker_node_test in tfjs-core contains a conditional require statement
    // that confuses the bundler of karma-typescript.
    ignore: ['./worker_node_test']
  }
};

const devConfig = {
  frameworks: ['jasmine', 'karma-typescript'],
  files: [
    {pattern: './node_modules/@babel/polyfill/dist/polyfill.js'},
    'src/setup_test.ts',
    {pattern: 'src/**/*.ts'},
  ],
  preprocessors: {'src/**/*.ts': ['karma-typescript']},
  karmaTypescriptConfig,
  reporters: ['dots', 'karma-typescript']
};

module.exports = function(config) {
  const args = [];
  if (config.grep) {
    args.push('--grep', config.grep);
  }
  if (config.flags) {
    args.push('--flags', config.flags);
  }
  let exclude = [];
  if (config.excludeTest != null) {
    exclude.push(config.excludeTest);
  }

  config.set({
    ...devConfig,
    exclude,
    port: 9876,
    colors: true,
    autoWatch: false,
    browsers: ['Chrome', 'chrome_webgpu'],
    singleRun: true,
    customLaunchers: {
      chrome_webgpu: {
        base: 'Chrome',
        flags: [
          '--enable-unsafe-webgpu', '--enable-dawn-features=disable_robustness'
        ],
      }
    },
    client: {jasmine: {random: false}, args: args}
  })
}
