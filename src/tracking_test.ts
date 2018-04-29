/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import {describeWithFlags} from './jasmine_util';
import {CPU_ENVS, WEBGL_ENVS} from './test_util';

describeWithFlags('time webgl', WEBGL_ENVS, () => {
  it('upload + compute', async () => {
    const a = tf.zeros([10, 10]);
    const time = await tf.time(() => a.square()) as tf.webgl.WebGLTimingInfo;
    expect(time.uploadWaitMs > 0);
    expect(time.downloadWaitMs === 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('upload + compute + dataSync', async () => {
    const a = tf.zeros([10, 10]);
    const time =
        await tf.time(() => a.square().dataSync()) as tf.webgl.WebGLTimingInfo;
    expect(time.uploadWaitMs > 0);
    expect(time.downloadWaitMs > 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('upload + compute + data', async () => {
    const a = tf.zeros([10, 10]);
    const time = await tf.time(async () => await a.square().data()) as
        tf.webgl.WebGLTimingInfo;
    expect(time.uploadWaitMs > 0);
    expect(time.downloadWaitMs > 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('preupload (not included) + compute + data', async () => {
    const a = tf.zeros([10, 10]);
    // Pre-upload a on gpu.
    a.square();
    const time = await tf.time(() => a.sqrt()) as tf.webgl.WebGLTimingInfo;
    // The tensor was already on gpu.
    expect(time.uploadWaitMs === 0);
    expect(time.downloadWaitMs === 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });
});

describeWithFlags('time cpu', CPU_ENVS, () => {
  it('simple upload', async () => {
    const a = tf.zeros([10, 10]);
    const time = await tf.time(() => a.square());
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });
});
