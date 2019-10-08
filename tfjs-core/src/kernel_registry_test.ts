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

import * as tf from './index';
import {ALL_ENVS, describeWithFlags} from './jasmine_util';
import {DataInfo} from './kernel_registry';

describeWithFlags('kernel_registry', ALL_ENVS, () => {
  it('register a kernel and call it', () => {
    let called = false;
    tf.registerKernel('MyKernel', tf.getBackend(), ({inputs, attrs}) => {
      expect(attrs.a).toBe(5);
      expect(inputs.x.shape).toEqual([2, 2]);
      expect(inputs.x.dtype).toBe('float32');
      called = true;
      return {dtype: 'float32', shape: [3, 3], dataId: {}};
    });

    const inputs = {x: tf.zeros([2, 2])};
    const attrs = {a: 5};
    const res = tf.engine().run('MyKernel', inputs, attrs) as DataInfo;

    expect(called).toBe(true);
    expect(res.dtype).toBe('float32');
    expect(res.shape).toEqual([3, 3]);

    tf.unregisterKernel('MyKernel', tf.getBackend());
  });

  it('errors when running non-existent kernel', () => {
    const inputs = {};
    const attrs = {};
    expect(() => tf.engine().run('DoesNotExist', inputs, attrs)).toThrowError();
  });

  it('errors when registering the same kernel twice', () => {
    tf.registerKernel('MyKernel', tf.getBackend(), () => {
      return null;
    });
    expect(() => tf.registerKernel('MyKernel', tf.getBackend(), () => {
      return null;
    })).toThrowError();

    tf.unregisterKernel('MyKernel', tf.getBackend());
  });
});
