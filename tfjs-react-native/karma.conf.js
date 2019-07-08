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
  bundlerOptions: {
    // Start from test files to control what karma typescript loads
    // and ensure that environment setup happens appropriately.
    entrypoints: /_test\.(ts)$/,
    // Mock react native functionality to enable unit tests in the browser.
    resolve: {
      alias: {
        'react-native': './test/utils/react_native_mock.ts',
        '@react-native-community/async-storage':
            './test/utils/async_storage_mock.ts',
      }
    }
  }
};

const baseConfig = {
  frameworks: ['jasmine', 'karma-typescript'],
  files: [
    './src/**/*.ts',
    './test/**/*.ts',
  ],
  preprocessors: {
    'src/**/*.ts': ['karma-typescript'],
    'test/**/*.ts': ['karma-typescript'],
  },
  karmaTypescriptConfig,
  reporters: ['verbose', 'karma-typescript'],
};

module.exports = function(config) {
  const args = [];
  if (config.grep) {
    args.push('--grep', config.grep);
  }

  config.set({
    ...baseConfig,
    basePath: '',
    frameworks: ['jasmine', 'karma-typescript'],
    preprocessors: {'**/*.ts': ['karma-typescript']},
    karmaTypescriptConfig,
    reporters: ['progress', 'karma-typescript'],
    port: 9876,
    colors: true,
    autoWatch: false,
    browsers: ['Chrome'],
    singleRun: true,
    client: {
      jasmine: {random: false},
      args: args,
    },
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
