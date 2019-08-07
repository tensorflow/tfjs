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
  reports: {},
  // Disable coverage reports and instrumentation by default for tests
  coverageOptions: {instrumentation: false},
  reports: {},
  bundlerOptions: {
    sourceMap: true,
    // Start from test files to control what karma typescript loads
    // and ensure that environment setup happens appropriately.
    entrypoints: /_test\.(ts)$/,
    // Mock react native functionality to enable unit tests in the browser.
    resolve: {
      alias: {
        'react-native': './src/test_utils/react_native_mock.ts',
        '@react-native-community/async-storage':
            './src/test_utils/async_storage_mock.ts',
        'expo-gl': './src/test_utils/gl_view_mock.ts',
      }
    }
  }
};

const baseConfig = {
  frameworks: ['jasmine', 'karma-typescript'],
  files: [
    './src/**/*.ts',
  ],
  preprocessors: {
    'src/**/*.ts': ['karma-typescript'],
  },
  karmaTypescriptConfig,
  reporters: ['verbose', 'karma-typescript'],
};

const browserstackConfig = {
  ...baseConfig,
  reporters: ['dots'],
  singleRun: true,
  hostname: 'bs-local.com',
  browserStack: {
    username: process.env.BROWSERSTACK_USERNAME,
    accessKey: process.env.BROWSERSTACK_KEY
  },
  captureTimeout: 120000,
  reportSlowerThan: 500,
  browserNoActivityTimeout: 180000,
};

module.exports = function(config) {
  const args = [];

  if (config.grep) {
    args.push('--grep', config.grep);
  }

  let extraConfig = config.browserstack ? browserstackConfig : baseConfig;

  config.set({
    basePath: '',
    karmaTypescriptConfig,
    reporters: ['progress', 'karma-typescript'],
    port: 9876,
    colors: true,
    autoWatch: false,
    browsers: ['Chrome'],
    client: {
      jasmine: {random: false},
      args: args,
    },
    ...extraConfig,
    customLaunchers: {
      // For browserstack configs see:
      // https://www.browserstack.com/automate/node
      bs_chrome_mac: {
        base: 'BrowserStack',
        browser: 'chrome',
        browser_version: 'latest',
        os: 'OS X',
        os_version: 'High Sierra'
      },
    }
  })
}
