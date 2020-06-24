/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
  tsconfig: 'tsconfig.json'
};

module.exports = function(config) {
  const args = [];
  if (config.grep) {
    args.push('--grep', config.grep);
  }
  if (config.firebaseKey) {
    args.push('--firebaseKey', config.firebaseKey);
  }
  if (config.nightly) {
    args.push('--nightly');
  }
  if (config.browsers) {
    args.push('--browsers', config.browsers);
  }
  if (config.hashes) {
    args.push('--hashes', config.hashes);
  }

  config.set({
    frameworks: ['jasmine', 'karma-typescript'],
    files: [{pattern: '*.ts'}],
    include: ['*.ts'],
    exclude: ['./run_node_tests.ts'],
    preprocessors: {
      '**/*.ts': ['karma-typescript'],  // *.tsx for React Jsx
    },
    karmaTypescriptConfig,
    reporters: ['progress', 'karma-typescript'],
    browsers: ['Chrome'],
    browserStack: {
      username: process.env.BROWSERSTACK_USERNAME,
      accessKey: process.env.BROWSERSTACK_KEY
    },
    captureTimeout: 10000000,
    browserDisconnectTimeout: 10000000,
    browserDisconnectTolerance: 10,
    reportSlowerThan: 500,
    browserNoActivityTimeout: 10000000,
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
      }
    },
    client: {jasmine: {random: false}, args: args}
  });
};
