/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

module.exports = function(config) {
  config.set({
    frameworks: ['jasmine', 'karma-typescript'],
    files: [{pattern: 'src/**/*.ts'}],
    exclude: [
      'src/docs/**/*.ts'
    ],
    preprocessors: {
      'src/**/*.ts': ['karma-typescript'],  // *.tsx for React Jsx
      'src/**/*.js': ['karma-typescript'],  // *.tsx for React Jsx
    },
    karmaTypescriptConfig: {
      tsconfig: 'tsconfig.json',
      compilerOptions: {
        allowJs: true,
        declaration: false,
        module: 'commonjs'
      },
      coverageOptions: {instrumentation: false},
      reports: {} // Do not produce coverage html.
    },
    reporters: ['progress', 'karma-typescript'],
    browsers: ['Chrome'],
    browserStack: {
      username: process.env.BROWSERSTACK_USERNAME,
      accessKey: process.env.BROWSERSTACK_KEY
    },
    reportSlowerThan: 500,
    browserNoActivityTimeout: 30000,
    customLaunchers: {
      bs_chrome_mac: {
        base: 'BrowserStack',
        browser: 'chrome',
        browser_version: 'latest',
        os: 'OS X',
        os_version: 'Sierra'
      },
      bs_firefox_mac: {
        base: 'BrowserStack',
        browser: 'firefox',
        // TODO(cais): Change it back to 'latest' once the ongoing instability
        // with regard to OS X and FireFox is resolved on BrowserStack:
        // https://github.com/tensorflow/tfjs/issues/1620
        browser_version: '66.0',
        os: 'OS X',
        os_version: 'Sierra'
      }
    },
    client: {
      args: ['--grep', config.grep || '']
    }
  });
};
