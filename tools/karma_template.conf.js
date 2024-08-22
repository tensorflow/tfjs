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

const browserstackConfig = {
  // Browserstack only supports a certain range of ports for safari.
  // Karma will automatically use the next available port if the
  // chosen one is in use. Starting at 9200 gives us the largest
  // range of ports (9200 - 9400).
  // https://www.browserstack.com/question/39572
  port: 9200,
};

// Select Chrome or ChromeHeadless based on the value of the --//:headless flag.
const CHROME = TEMPLATE_headless ? 'ChromeHeadless' : 'Chrome';

const CUSTOM_LAUNCHERS = {
  // For browserstack configs see:
  // https://www.browserstack.com/automate/node
  bs_chrome_mac: {
    base: 'BrowserStack',
    browser: 'chrome',
    browser_version: 'latest',
    os: 'OS X',
    os_version: 'High Sierra',
    flags: [
      // For tfjs-data
      '--autoplay-policy=no-user-gesture-required',
    ],
  },
  bs_firefox_mac: {
    base: 'BrowserStack',
    browser: 'firefox',
    browser_version: '90',
    os: 'OS X',
    os_version: 'High Sierra'
  },
  bs_safari_mac: {
    base: 'BrowserStack',
    browser: 'safari',
    browser_version: 'latest',
    os: 'OS X',
    os_version: 'Mojave'
  },
  bs_ios_12: {
    base: 'BrowserStack',
    device: 'iPhone XS',
    os: 'ios',
    os_version: '12.3',
    real_mobile: true
  },
  bs_ios_17: {
    base: 'BrowserStack',
    device: 'iPhone 15 Pro Max',
    os: 'ios',
    os_version: '17',
    real_mobile: true
  },
  bs_android_10: {
    base: 'BrowserStack',
    device: 'Google Pixel 4 XL',
    os: 'android',
    os_version: '10.0',
    real_mobile: true
  },
  win_10_chrome: {
    base: 'BrowserStack',
    browser: 'chrome',
    browser_version: '104.0',
    os: 'Windows',
    os_version: '10',
    flags: [
      // For tfjs-data
      '--autoplay-policy=no-user-gesture-required',
    ],
  },
  bs_chrome_mac_webgpu: {
    base: 'BrowserStack',
    browser: 'chrome',
    browser_version: 'latest',
    os: 'OS X',
    os_version: 'High Sierra',
    flags: [
      '--enable-unsafe-webgpu',  // Can be removed after WebGPU release
      '--use-webgpu-adapter=swiftshader',

      // https://github.com/tensorflow/tfjs/issues/7631
      '--disable-vulkan-fallback-to-gl-for-testing',
    ],
  },
  chrome_with_swift_shader: {
    base: CHROME,
    flags: ['--blacklist-accelerated-compositing', '--blacklist-webgl']
  },
  chrome_autoplay: {
    base: CHROME,
    flags: [
      '--autoplay-policy=no-user-gesture-required',
      '--no-sandbox',
    ],
  },
  chrome_webgpu_linux: {
    base: 'ChromeCanary',
    flags: [
      '--enable-features=Vulkan',
      '--enable-unsafe-webgpu',
      '--disable-dawn-features=disallow_unsafe_apis',
    ]
  },
  chrome_webgpu: {
    base: 'ChromeCanary',
    flags: [
      '--disable-dawn-features=disallow_unsafe_apis',
      '--no-sandbox',
    ]
  },
  chrome_debugging: {
    base: 'Chrome',
    flags: ['--remote-debugging-port=9333'],
  },
  chrome_no_sandbox: {
    base: CHROME,
    flags: ['--no-sandbox'],
  }
};

module.exports = function(config) {
  console.log(`Running with arguments ${TEMPLATE_args.join(' ')}`);
  let browser = 'TEMPLATE_browser';
  let extraConfig = {};
  const browserLauncher = CUSTOM_LAUNCHERS[browser];
  if (browser) {
    if (!browserLauncher) {
      throw new Error(`Missing launcher for ${browser}`);
    }
    extraConfig.browsers = [browser];
  } else {
    // Use no sandbox by default. This has better support on MacOS.
    extraConfig.browsers = ['chrome_no_sandbox'];
  }
  if (browserLauncher?.base === 'BrowserStack') {
    const username = process.env.BROWSERSTACK_USERNAME;
    const accessKey = process.env.BROWSERSTACK_KEY;
    if (!username) {
      console.error(
          'No browserstack username found. Please set the' +
          ' environment variable "BROWSERSTACK_USERNAME" to your' +
          ' browserstack username');
    }
    if (!accessKey) {
      console.error(
          'No browserstack access key found. Please set the' +
          ' environment variable "BROWSERSTACK_KEY" to your' +
          ' browserstack access key');
    }
    if (!username || !accessKey) {
      process.exit(1);
    }
    if (browserLauncher.browser === 'safari' || browserLauncher.os === 'ios') {
      // This is necessary for non-flaky Safari tests. They usually pass just
      // fine without it, but sometimes, Safari will fail to connect to Karma.
      // If you want to remove this, prove that it's not flaky by running
      // bazel test //tfjs-core/src:bs_safari_mac_from_pixels_worker_test
      // --runs_per_test=100
      extraConfig.hostname = 'bs-local.com';
    }

    Object.assign(extraConfig, browserstackConfig);
    extraConfig.browserStack = {
      username: process.env.BROWSERSTACK_USERNAME,
      accessKey: process.env.BROWSERSTACK_KEY,
      timeout: 900,  // Seconds
      tunnelIdentifier: `tfjs_${Date.now()}_${Math.floor(Math.random() * 1000)}`
    };
  }

  config.set({
    reporters: [
      'kjhtml',
      'jasmine-order',
    ],
    frameworks: ['jasmine'],
    plugins: [
      require('karma-jasmine'),
      require('karma-jasmine-html-reporter'),
      require('karma-jasmine-order-reporter'),
    ],
    captureTimeout: 3e5,
    reportSlowerThan: 500,
    browserNoActivityTimeout: 3e5,
    browserDisconnectTimeout: 3e5,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 1.2e5,
    ...extraConfig,
    customLaunchers: CUSTOM_LAUNCHERS,
    client: {
      args: TEMPLATE_args,
      jasmine: {
        random: TEMPLATE_jasmine_random,
        seed: 'TEMPLATE_jasmine_seed',
      },
    },
  });
}
