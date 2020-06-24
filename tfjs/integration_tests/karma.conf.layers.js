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
  if (config.log) {
    args.push('--log', config.log);
  }
  if (config.hashes) {
    args.push('--hashes', config.hashes);
  }
  if (config.nightly) {
    args.push('--nightly');
  }

  config.set({
    frameworks: ['jasmine', 'karma-typescript'],
    files: [
      {pattern: 'models/*.ts'}, {pattern: 'firestore.ts'}, {
        pattern: 'data/**/*',
        watched: false,
        included: false,
        served: true,
        nocache: true
      }
    ],
    include: ['models/*.ts'],
    exclude: ['models/validation.ts'],
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
    captureTimeout: 120000,
    browserNoActivityTimeout: 180000,
    customLaunchers: {
      bs_chrome_mac: {
        base: 'BrowserStack',
        browser: 'chrome',
        browser_version: 'latest',
        os: 'OS X',
        os_version: 'High Sierra'
      }
    },
    client: {jasmine: {random: false}, args: args}
  });
};
