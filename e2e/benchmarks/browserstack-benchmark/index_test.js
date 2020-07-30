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

describe('constructBrowserTree', () => {
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

  const expectedTree = {
    'OS X': {'High Sierra': {'firefox': {'70.0': {'null': mac}}}},
    'ios': {'11.0': {'ios': {'null': {'iPhone X': iphoneX}}}}
  };

  it('constructs a tree', () => {
    const browsersArray = [mac, iphoneX];
    expect(constructBrowserTree(browsersArray)).toEqual(expectedTree);
  });

  it('warns when finding duplicate nodes', () => {
    const browsersArray = [mac, iphoneX, iphoneX];
    spyOn(console, 'warn');
    expect(constructBrowserTree(browsersArray)).toEqual(expectedTree);
    expect(console.warn).toHaveBeenCalled();
  });
});
