/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
  compilerOptions: {allowJs: true, declaration: false},
  bundlerOptions: {
    sourceMap: true,
    acornOptions: {ecmaVersion: 8},
  },
  // Disable coverage reports and instrumentation by default for tests
  coverageOptions: {instrumentation: false},
  reports: {},
};

const devConfig = {
  frameworks: ['jasmine', 'karma-typescript'],
  files: [
    {pattern: './node_modules/@babel/polyfill/dist/polyfill.js'},
    'src/setup_test.ts',
    {pattern: 'src/**/*.ts'},
    'src/tflite_web_api_client.js',
  ],
  exclude: [
    'src/index.ts',
  ],
  preprocessors: {
    'src/tflite_web_api_client.js': ['karma-typescript'],
    '**/*.ts': ['karma-typescript'],
    '**/*.d.ts': ['karma-typescript'],
  },
  karmaTypescriptConfig,
  reporters: ['progress', 'karma-typescript']
};

const browserstackConfig = {
  ...devConfig,
  hostname: 'bs-local.com',
  singleRun: true,
  port: 9206
};

module.exports = function(config) {
  const args = [];
  if (config.grep) {
    args.push('--grep', config.grep);
  }
  if (config.flags) {
    args.push('--flags', config.flags);
  }

  let extraConfig = null;

  if (config.browserstack) {
    extraConfig = browserstackConfig;
  } else {
    extraConfig = devConfig;
  }

  config.set({
    ...extraConfig,
    basePath: '',
    // Redirect the request for the wasm file so karma can find it.
    proxies: {
      '/base/node_modules/karma-typescript/dist/client/tflite_web_api_cc.js':
          '/base/deps/tflite_web_api_cc.js',
      '/base/node_modules/karma-typescript/dist/client/tflite_web_api_cc_threaded.js':
          '/base/deps/tflite_web_api_cc_threaded.js',
      '/base/node_modules/karma-typescript/dist/client/tflite_web_api_cc_simd_threaded.js':
          '/base/deps/tflite_web_api_cc_simd_threaded.js',
    },
    browsers: ['Chrome'],
    browserStack: {
      username: process.env.BROWSERSTACK_USERNAME,
      accessKey: process.env.BROWSERSTACK_KEY,
      timeout: 1800,
      tunnelIdentifier:
          `tfjs_tflite_${Date.now()}_${Math.floor(Math.random() * 1000)}`
    },
    captureTimeout: 3e5,
    reportSlowerThan: 500,
    browserNoActivityTimeout: 3e5,
    browserDisconnectTimeout: 3e5,
    browserDisconnectTolerance: 3,
    browserSocketTimeout: 1.2e5,
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
        browser_version: 'latest',
        os: 'Windows',
        os_version: '10'
      },
    },
    client: {jasmine: {random: false}, args: args},
  })
}
