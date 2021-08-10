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
  port: 9896,
};

module.exports = function(config) {
  config.set({
    ...browserstackConfig,
    browserStack: {
      username: process.env.BROWSERSTACK_USERNAME,
      accessKey: process.env.BROWSERSTACK_KEY,
      timeout: 900, // Seconds
      tunnelIdentifier:
      `tfjs_${Date.now()}_${Math.floor(Math.random() * 1000)}`
    },
    captureTimeout: 3e5,
    reportSlowerThan: 500,
    browserNoActivityTimeout: 3e5,
    browserDisconnectTimeout: 3e5,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 1.2e5,
    browsers: ['TEMPLATE_browser'],
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
        // Latest Chrome on Windows has WebGL problems:
        // https://github.com/tensorflow/tfjs/issues/2272
        browser_version: '77.0',
        os: 'Windows',
        os_version: '10'
      },
      chrome_with_swift_shader: {
        base: 'Chrome',
        flags: ['--blacklist-accelerated-compositing', '--blacklist-webgl']
      },
      chrome_debugging:
      {base: 'Chrome', flags: ['--remote-debugging-port=9333']}
    }
  });
}
