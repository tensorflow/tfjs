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


const localIdentifier =
    `rn_integration_${Date.now()}_${Math.floor(Math.random() * 10000)}`;

const sharedCapabilities = Object.freeze({
  'browserstack.debug': true,
  'browserstack.local': true,
  'browserstack.networkLogs': true,
  'browserstack.localIdentifier': localIdentifier,
});


exports.config = {
  before: function() {
    require('ts-node').register({files: true})
  },

  // ====================
  // Runner Configuration
  // ====================
  runner: 'local',

  // =====================
  // Server Configurations
  // =====================
  hostname: 'hub-cloud.browserstack.com',
  port: 443,
  services: ['browserstack'],
  user: process.env.BROWSERSTACK_USERNAME,
  key: process.env.BROWSERSTACK_KEY,
  maxInstances: 5,
  // Bridge network requests to the machine that started the test session.
  browserstackLocal: true,
  browserstackLocalForcedStop: true,
  browserstackOpts: {
    // Create a local id to separate this instance of browserstack local from
    // ones that may be created by karma.
    localIdentifier,
  },

  // =====================
  // Test configuration
  // =====================
  specs: ['./test/**/*.ts'],
  exclude: [],
  capabilities: [
    Object.assign(
        {
          'app': 'deeplearnjs1/tfjs-rn-integration-android',
          'device': 'Samsung Galaxy S9 Plus',
          'os': 'android',
          'os_version': '9.0',
          'build': 'tfjs-react-native android',
          'name': 'React Native Integration Test (Android)',
        },
        sharedCapabilities),
  ],
  // Level of logging verbosity: trace | debug | info | warn | error | silent
  logLevel: 'warn',
  // If you only want to run your tests until a specific amount of tests have
  // failed use bail (default is 0 - don't bail, run all tests).
  bail: 0,
  // Default timeout for all waitFor* commands.
  waitforTimeout: 10000,
  // Default timeout in milliseconds for request
  connectionRetryTimeout: 180000,
  // Default request retries count
  connectionRetryCount: 3,
  framework: 'jasmine',
  reporters: ['spec'],
  jasmineNodeOpts: {
    defaultTimeoutInterval: 60000,
  },
}
