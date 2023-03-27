/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

import {ALL_ENVS, BROWSER_ENVS, describeWithFlags} from '../jasmine_util';

import {toPixelsNoCanvas, toPixelsTestCase, toPixelsWithCanvas} from './to_pixels_util_test';

describeWithFlags('toPixels no canvas, returns data', ALL_ENVS, () => {
  toPixelsTestCase(toPixelsNoCanvas, true);
});

describeWithFlags('toPixels canvas, returns void', BROWSER_ENVS, () => {
  toPixelsTestCase(toPixelsWithCanvas, false);
});

describeWithFlags('toPixels canvas, returns data', BROWSER_ENVS, () => {
  toPixelsTestCase(toPixelsWithCanvas, true);
});
