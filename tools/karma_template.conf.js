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
  hostname: 'bs-local.com',
  reporters: ['dots'],
  port: 9876,
};

const CUSTOM_LAUNCHERS = {
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
    browser_version: '90',
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
    os: 'ios',
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
    browser_version: '101.0',
    os: 'Windows',
    os_version: '10'
  },
  chrome_with_swift_shader: {
    base: 'Chrome',
    flags: ['--blacklist-accelerated-compositing', '--blacklist-webgl']
  },
  chrome_webgpu: {
    base: 'ChromeCanary',
    flags: [
      '--enable-unsafe-webgpu',
      '--disable-dawn-features=disallow_unsafe_apis'
    ]
  },
  chrome_debugging:
      {base: 'Chrome', flags: ['--remote-debugging-port=9333']}
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
  }
  if (browserLauncher?.base === 'BrowserStack') {
    const username = process.env.BROWSERSTACK_USERNAME;
    const accessKey = process.env.BROWSERSTACK_KEY;
    if (!username) {
      console.error('No browserstack username found. Please set the'
                    + ' environment variable "BROWSERSTACK_USERNAME" to your'
                    + ' browserstack username');
    }
    if (!accessKey) {
      console.error('No browserstack access key found. Please set the'
                    + ' environment variable "BROWSERSTACK_KEY" to your'
                    + ' browserstack access key');
    }
    if (!username || !accessKey) {
      process.exit(1);
    }

    Object.assign(extraConfig, browserstackConfig);
    extraConfig.browserStack = {
      username: process.env.BROWSERSTACK_USERNAME,
      accessKey: process.env.BROWSERSTACK_KEY,
      timeout: 900,  // Seconds
      tunnelIdentifier:
      `tfjs_${Date.now()}_${Math.floor(Math.random() * 1000)}`
    };
  }

  config.set({
    captureTimeout: 3e5,
    reportSlowerThan: 500,
    browserNoActivityTimeout: 3e5,
    browserDisconnectTimeout: 3e5,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 1.2e5,
    ...extraConfig,
    customLaunchers: CUSTOM_LAUNCHERS,
    client: {args: TEMPLATE_args},
  });
}
