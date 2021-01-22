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
    acornOptions: {
      ecmaVersion: 8,
    },
    transforms: [
      require('karma-typescript-es6-transform')({
        presets: [[
          'env', {
            targets: {
              browsers: [
                'last 10 Chrome versions',
                'last 10 Firefox versions',
                'last 5 Safari versions',
              ]
            }
          }
        ]]
      }),
    ]
  }
};

// Enable coverage reports and instrumentation under KARMA_COVERAGE=1 env
const coverageEnabled = !!process.env.KARMA_COVERAGE;
if (coverageEnabled) {
  karmaTypescriptConfig.coverageOptions.instrumentation = true;
  karmaTypescriptConfig.coverageOptions.exclude = /_test\.ts.*/;
  karmaTypescriptConfig.reports = {html: 'coverage', 'text-summary': ''};
}

module.exports = function(config) {
  const args = [];

  if (config.grep) {
    args.push('--grep', config.grep);
  }

  config.set({
    frameworks: ['jasmine', 'karma-typescript'],
    files: ['src/setup_test.ts', 'src/**/*.ts*'],
    exclude: ['src/types/**'],
    preprocessors: {
      '**/*.ts': ['karma-typescript'],
      '**/*.tsx': ['karma-typescript'],
      '**/*.d.ts': ['karma-typescript'],
    },
    karmaTypescriptConfig,
    reporters: ['progress', 'karma-typescript'],
    browsers: ['Chrome'],
    port: 9836,
    browserStack: {
      username: process.env.BROWSERSTACK_USERNAME,
      accessKey: process.env.BROWSERSTACK_KEY,
      tunnelIdentifier:
          `tfjs_vis_${Date.now()}_${Math.floor(Math.random() * 1000)}`
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
    },
    client: {
      jasmine: {random: false},
      args: args,
    }
  });
};
