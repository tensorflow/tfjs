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

/**
 * The unit tests in this file can be run by opening `SpecRunner.html` in
 * browser.
 */

describe('getTabId', () => {
  const mac = {
    base: 'BrowserStack',
    browser: 'firefox',
    browser_version: '70.0',
    os: 'OS X',
    os_version: 'High Sierra',
    device: 'null'
  };
  const iphoneX = {
    base: 'BrowserStack',
    browser: 'ios',
    browser_version: 'null',
    os: 'ios',
    os_version: '11.0',
    device: 'iPhone X'
  };

  it('gives different names for the same browser configs', () => {
    const name1 = getTabId(mac);
    const name2 = getTabId(mac);
    expect(name1).not.toBe(name2);
  });

  it('for mobile devices, uses device name as part of the tab name', () => {
    const mobileName = getTabId(iphoneX);
    expect(mobileName).toContain(iphoneX.device);
  });

  it('for desktop devices, uses OS name as part of the tab name', () => {
    const desktopName = getTabId(mac);
    expect(desktopName).toContain(mac.os);
    expect(desktopName).toContain(mac.os_version);
  });
});
