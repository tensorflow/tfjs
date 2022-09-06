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
 * The unit tests in this file can be run by opening `./SpecRunner.html` in
 * browser.
 */

describe('benchmark multiple browsers', () => {
  const browsersList = [
    {
      'os': 'OS X',
      'os_version': 'Catalina',
      'browser': 'chrome',
      'device': null,
      'browser_version': '84.0'
    },
    {
      'os': 'android',
      'os_version': '9.0',
      'browser': 'android',
      'device': 'Samsung Galaxy Note 10 Plus',
      'browser_version': null,
      'real_mobile': true
    },
    {
      'os': 'Windows',
      'os_version': '10',
      'browser': 'chrome',
      'device': null,
      'browser_version': '84.0',
      'real_mobile': null
    }
  ];
  const benchmark = {model: 'mobilenet_v2', numRuns: 1, backend: 'cpu'};
  const browsers = {};

  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;

    // Populate `browsers`.
    for (const browserConfig of browsersList) {
      const tabId = getTabId(browserConfig);
      browsers[tabId] = browserConfig;
    }
  });

  it('the number of received results is equal to the number of browsers',
     done => {
       socket.emit('run', {benchmark, browsers});

       benchmarkResults = [];
       socket.on('benchmarkComplete', benchmarkResult => {
         expect(benchmarkResult.error).toBeUndefined();
         benchmarkResults.push(benchmarkResult);
         if (benchmarkResults.length === browsersList.length) {
           done();
         }
       });
     });
});
