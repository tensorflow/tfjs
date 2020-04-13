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

const karmaTypescriptConfig = {
  tsconfig: 'tsconfig.json',
  // Disable coverage reports and instrumentation by default for tests
  coverageOptions: {instrumentation: false},
  reports: {},
  bundlerOptions: {
    sourceMap: true,
    // Ignore the import of the `worker_threads` package used in a core test
    // meant to run in node.
    exclude: ['worker_threads'],
    // worker_node_test in tfjs-core contains a conditional require statement
    // that confuses the bundler of karma-typescript.
    ignore: ['./worker_node_test']
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
  let exclude = [];
  if (config.excludeTest != null) {
    exclude.push(config.excludeTest);
  }

  config.set({
    frameworks: ['jasmine', 'karma-typescript'],
    files: [
      'src/setup_test.ts',
      {pattern: 'src/**/*.ts'},
    ],
    preprocessors: {'**/*.ts': ['karma-typescript']},
    karmaTypescriptConfig,
    reporters: ['dots', 'karma-typescript'],
    exclude,
    colors: true,
    autoWatch: false,
    browsers: ['Chrome'],
    singleRun: true,
    client: {jasmine: {random: false}, args: args},
    browserStack: {
      username: process.env.BROWSERSTACK_USERNAME,
      accessKey: process.env.BROWSERSTACK_KEY
    },
    captureTimeout: 120000,
    reportSlowerThan: 500,
    browserNoActivityTimeout: 240000,
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
      bs_android_9: {
        base: 'BrowserStack',
        device: 'Google Pixel 3 XL',
        os: 'android',
        os_version: '9.0',
        real_mobile: true
      },
      win_10_chrome: {
        base: 'BrowserStack',
        browser: 'chrome',
        // Latest Chrome on Windows has WebGL problems:
        // https://github.com/tensorflow/tfjs/issues/2272
        browser_version: '77.0',
        os: 'Windows',
        os_version: '10'
      },
    },
  })
}
