/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {PlatformReactNative} from './platform_react_native';
import {RN_ENVS} from './test_env_registry';

describeWithFlags('PlatformReactNative', RN_ENVS, () => {
  it('tf.util.fetch calls platform.fetch', async () => {
    const platform = new PlatformReactNative();
    tf.setPlatform('rn-test-platform', platform);

    spyOn(platform, 'fetch');

    await tf.util.fetch('test/url', {method: 'GET'});
    expect(platform.fetch).toHaveBeenCalledWith('test/url', {method: 'GET'});
  });
});
