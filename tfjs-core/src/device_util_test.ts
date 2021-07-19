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

import * as device_util from './device_util';
import {ALL_ENVS, describeWithFlags} from './jasmine_util';

describeWithFlags('isMobile', ALL_ENVS, () => {
  it('should not fail when navigator is set', () => {
    expect(() => device_util.isMobile()).not.toThrow();
  });
  it('identifies react native as a mobile device', () => {
    expect(device_util.isMobile(
      {product: 'ReactNative'} as Navigator)).toEqual(true);
  });
});
