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
  tsconfig: 'tsconfig.json',
  compilerOptions: {allowJs: true, declaration: false},
  bundlerOptions: {
    sourceMap: true,
    // Ignore the import of the `worker_threads` package used in a core test
    // meant to run in node.
    exclude: ['worker_threads']
  },
  // Disable coverage reports and instrumentation by default for tests
  coverageOptions: {instrumentation: false},
  reports: {},
  include: ['src/', 'wasm-out/']
};

module.exports = function(config) {
  const args = [];
  if (config.grep) {
    args.push('--grep', config.grep);
  }
  if (config.flags) {
    args.push('--flags', config.flags);
  }
  config.set({
    basePath: '',
    frameworks: ['jasmine', 'karma-typescript'],
    files: [
      // Setup the environment for the tests.
      'src/setup_test.ts',
      // Serve the wasm file as a static resource.
      {pattern: 'wasm-out/**/*.wasm', included: false},
      // Import the generated js library from emscripten.
      {pattern: 'wasm-out/**/*.js'},
      // Import the rest of the sources.
      {pattern: 'src/**/*.ts'},
    ],
    preprocessors: {
      'wasm-out/**/*.js': ['karma-typescript'],
      '**/*.ts': ['karma-typescript']
    },
    karmaTypescriptConfig,
    // Redirect the request for the wasm file so karma can find it.
    proxies: {
      '/base/node_modules/karma-typescript/dist/client/tfjs-backend-wasm.wasm':
          '/base/wasm-out/tfjs-backend-wasm.wasm',
    },
    reporters: ['progress', 'karma-typescript'],
    port: 9876,
    colors: true,
    autoWatch: true,
    browsers: ['Chrome'],
    singleRun: false,
    client: {jasmine: {random: false}, args: args}
  })
}
