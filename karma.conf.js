/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
  bundlerOptions: {sourceMap: true}
};

// Enable coverage reports and instrumentation under KARMA_COVERAGE=1 env
const coverageEnabled = !!process.env.KARMA_COVERAGE;
if (coverageEnabled) {
  karmaTypescriptConfig.coverageOptions.instrumentation = true;
  karmaTypescriptConfig.coverageOptions.exclude = /_test\.ts$/;
  karmaTypescriptConfig.reports = {html: 'coverage', 'text-summary': ''};
}

const devConfig = {
  frameworks: ['jasmine', 'karma-typescript'],
  files: ['src/setup_test.ts', {pattern: 'src/**/*.ts'}],
  exclude: [
    'src/tests.ts',
    'src/worker_node_test.ts',
    'src/worker_test.ts',
    'src/test_node.ts',
    'src/test_async_backends.ts',
  ],
  preprocessors: {'**/*.ts': ['karma-typescript']},
  karmaTypescriptConfig,
  reporters: ['dots', 'karma-typescript'],
};

const browserstackConfig = {
  frameworks: ['browserify', 'jasmine'],
  files: ['dist/setup_test.js', {pattern: 'dist/**/*_test.js'}],
  exclude: [
    'dist/worker_node_test.js',
    'dist/worker_test.js',
    'dist/test_node.js',
    'dist/test_async_backends.js',
  ],
  preprocessors: {'dist/**/*_test.js': ['browserify']},
  browserify: {debug: false},
  reporters: ['dots'],
  singleRun: true,
  hostname: 'bs-local.com',
};

const webworkerConfig = {
  ...browserstackConfig,
  files: [
    'dist/setup_test.js',
    'dist/worker_test.js',
    // Serve dist/tf-core.min.js as a static resource, but do not include in the
    // test runner
    {pattern: 'dist/tf-core.min.js', included: false},
  ],
  exclude: [],
  port: 12345
};

module.exports = function(config) {
  const args = [];
  // If no test environment is set unit tests will run against all registered
  // test environments.
  if (config.testEnv) {
    args.push('--testEnv', config.testEnv);
  }
  if (config.grep) {
    args.push('--grep', config.grep);
  }
  if (config.flags) {
    args.push('--flags', config.flags);
  }


  let extraConfig = null;

  if (config.worker) {
    extraConfig = webworkerConfig;
  } else if (config.browserstack) {
    extraConfig = browserstackConfig;
  } else {
    extraConfig = devConfig;
  }


  config.set({
    ...extraConfig,
    browsers: ['Chrome'],
    browserStack: {
      username: process.env.BROWSERSTACK_USERNAME,
      accessKey: process.env.BROWSERSTACK_KEY
    },
    captureTimeout: 120000,
    reportSlowerThan: 500,
    browserNoActivityTimeout: 180000,
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
      win_10_chrome: {
        base: 'BrowserStack',
        browser: 'chrome',
        browser_version: 'latest',
        os: 'Windows',
        os_version: '10'
      },
      chrome_with_swift_shader: {
        base: 'Chrome',
        flags: ['--blacklist-accelerated-compositing', '--blacklist-webgl']
      },
      chrome_debugging:
          {base: 'Chrome', flags: ['--remote-debugging-port=9333']}
    },
    client: {jasmine: {random: false}, args: args}
  });
};
