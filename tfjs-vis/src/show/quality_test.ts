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

import {showPerClassAccuracy} from './quality';

describe('perClassAccuracy', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
  });

  it('renders perClassAccuracy', async () => {
    const container = {name: 'Test'};
    const acc = [
      {accuracy: 0.5, count: 10},
      {accuracy: 0.8, count: 10},
    ];

    const labels = ['cat', 'dog'];
    await showPerClassAccuracy(container, acc, labels);
    expect(document.querySelectorAll('table').length).toBe(1);
  });

  it('renders perClassAccuracy without explicit labels', async () => {
    const container = {name: 'Test'};
    const acc = [
      {accuracy: 0.5, count: 10},
      {accuracy: 0.8, count: 10},
    ];
    await showPerClassAccuracy(container, acc);
    expect(document.querySelectorAll('table').length).toBe(1);
  });
});
