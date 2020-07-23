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

const localRunConfig = {
  reporters: ['progress'],
  plugins: ['karma-jasmine', 'karma-chrome-launcher'],
  browsers: ['Chrome']
};

const browserstackConfig = {
  hostname: 'bs-local.com',
  plugins: ['karma-jasmine', 'karma-browserstack-launcher'],
  reporters: ['progress', 'BrowserStack'],
  browserStack: {
    username: process.env.BROWSERSTACK_USERNAME,
    accessKey: process.env.BROWSERSTACK_ACCESS_KEY,
    apiClientEndpoint: 'https://api.browserstack.com'
  },

  customLaunchers: {
    bs_chrome_mac: {
      base: 'BrowserStack',
      browser: 'chrome',
      browser_version: '84.0',
      os: 'OS X',
      os_version: 'Catalina',
    },
    bs_firefox_mac: {
      base: 'BrowserStack',
      browser: 'firefox',
      browser_version: '70.0',
      os: 'OS X',
      os_version: 'Catalina',
    },
    bs_safari_mac: {
      base: 'BrowserStack',
      browser: 'Safari',
      browser_version: '13.1',
      os: 'OS X',
      os_version: 'Catalina',
    }
  },

  browsers: ['bs_chrome_mac', 'bs_firefox_mac', 'bs_safari_mac'],
};


module.exports = function(config) {
  let extraConfig = null;
  if (config.browserstack) {
    extraConfig = browserstackConfig;
  } else {
    extraConfig = localRunConfig;
  }

  config.set({
    ...extraConfig,
    frameworks: ['jasmine'],
    files: [
      'https://unpkg.com/@tensorflow/tfjs-core@latest/dist/tf-core.js',
      'https://unpkg.com/@tensorflow/tfjs-backend-cpu@latest/dist/tf-backend-cpu.js',
      'https://unpkg.com/@tensorflow/tfjs-backend-webgl@latest/dist/tf-backend-webgl.js',
      'https://unpkg.com/@tensorflow/tfjs-layers@latest/dist/tf-layers.js',
      'https://unpkg.com/@tensorflow/tfjs-converter@latest/dist/tf-converter.js',
      'https://unpkg.com/@tensorflow/tfjs-backend-wasm@latest/dist/tf-backend-wasm.js',
      'util.js', 'benchmark_util.js', 'benchmark_models.js'
    ],
    preprocessors: {},
    singleRun: true,
    captureTimeout: 3e5,
    reportSlowerThan: 500,
    browserNoActivityTimeout: 3e5,
    browserDisconnectTimeout: 3e5,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 1.2e5,
    client: {jasmine: {random: false}},

    // The following configurations are generated by karma
    port: 9876,
    colors: true,
    logLevel: config.LOG_INFO,
    autoWatch: false,
    concurrency: Infinity
  })
}
