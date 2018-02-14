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

import * as dl from './index';
import {CPU_ENVS, describeWithFlags, WEBGL_ENVS} from './test_util';

describeWithFlags('time webgl', WEBGL_ENVS, () => {
  it('upload + compute', async () => {
    const a = dl.zeros([10, 10]);
    const time = await dl.time(() => a.square()) as dl.WebGLTimingInfo;
    expect(time.uploadWaitMs > 0);
    expect(time.downloadWaitMs === 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('upload + compute + dataSync', async () => {
    const a = dl.zeros([10, 10]);
    const time =
        await dl.time(() => a.square().dataSync()) as dl.WebGLTimingInfo;
    expect(time.uploadWaitMs > 0);
    expect(time.downloadWaitMs > 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('upload + compute + data', async () => {
    const a = dl.zeros([10, 10]);
    const time = await dl.time(async () => await a.square().data()) as
        dl.WebGLTimingInfo;
    expect(time.uploadWaitMs > 0);
    expect(time.downloadWaitMs > 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });

  it('preupload (not included) + compute + data', async () => {
    const a = dl.zeros([10, 10]);
    // Pre-upload a on gpu.
    a.square();
    const time = await dl.time(() => a.sqrt()) as dl.WebGLTimingInfo;
    // The tensor was already on gpu.
    expect(time.uploadWaitMs === 0);
    expect(time.downloadWaitMs === 0);
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });
});

describeWithFlags('time cpu', CPU_ENVS, () => {
  it('simple upload', async () => {
    const a = dl.zeros([10, 10]);
    const time = await dl.time(() => a.square());
    expect(time.kernelMs > 0);
    expect(time.wallMs >= time.kernelMs);
  });
});
