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
    sourceMap: true,  // Process any non es5 code through
                      // karma-typescript-es6-transform (babel)
    acornOptions: {ecmaVersion: 8},
    transforms: [
      require('karma-typescript-es6-transform')({
        presets: [
          // ensure we get es5 by adding IE 11 as a target
          ['@babel/env', {'targets': {'ie': '11'}, 'loose': true}]
        ]
      }),
    ]
  }
};

module.exports = function(config) {
  const args = [];
  if (config.testEnv) {
    args.push('--testEnv', config.testEnv);
  }
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
      {pattern: './node_modules/@babel/polyfill/dist/polyfill.js'},
      'src/test_browser.ts',
      {pattern: 'src/**/*.ts'},
    ],
    exclude: ['src/test_node.ts'],
    preprocessors: {'**/*.ts': ['karma-typescript']},
    karmaTypescriptConfig,
    reporters: ['progress', 'karma-typescript'],
    port: 9866,
    colors: true,
    browsers: ['Chrome'],
    client: {jasmine: {random: false}, args: args},
    browserStack: {
      username: process.env.BROWSERSTACK_USERNAME,
      accessKey: process.env.BROWSERSTACK_KEY,
      tunnelIdentifier:
          `tfjs_automl_${Date.now()}_${Math.floor(Math.random() * 1000)}`
    },
    captureTimeout: 120000,
    reportSlowerThan: 500,
    browserNoActivityTimeout: 180000,
    customLaunchers: {
      bs_chrome_mac: {
        base: 'BrowserStack',
        browser: 'chrome',
        browser_version: 'latest',
        os: 'OS X',
        os_version: 'High Sierra'
      },
      bs_firefox_mac: {
        base: 'BrowserStack',
        browser: 'firefox',
        browser_version: 'latest',
        os: 'OS X',
        os_version: 'High Sierra'
      },
      bs_safari_mac: {
        base: 'BrowserStack',
        browser: 'safari',
        browser_version: 'latest',
        os: 'OS X',
        os_version: 'High Sierra'
      },
      bs_ios_11: {
        base: 'BrowserStack',
        device: 'iPhone X',
        os: 'iOS',
        os_version: '11.0',
        real_mobile: true
      },
    }
  })
}
