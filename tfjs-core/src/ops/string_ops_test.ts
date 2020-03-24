/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysEqual} from '../test_util';

const txtArr = [
  'Hello TensorFlow.js!', '𝌆', 'Pre\u2014trained models with Base64 ops\u002e',
  'how about these? 🌍💻🍕', 'https://www.tensorflow.org/js', 'àβÇdéf',
  '你好, 世界', `Build, train, & deploy
ML models in JS`
];
const urlSafeB64 = [
  'SGVsbG8gVGVuc29yRmxvdy5qcyE', '8J2Mhg',
  'UHJl4oCUdHJhaW5lZCBtb2RlbHMgd2l0aCBCYXNlNjQgb3BzLg',
  'aG93IGFib3V0IHRoZXNlPyDwn4yN8J-Su_CfjZU',
  'aHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvanM', 'w6DOssOHZMOpZg',
  '5L2g5aW9LCDkuJbnlYw', 'QnVpbGQsIHRyYWluLCAmIGRlcGxveQpNTCBtb2RlbHMgaW4gSlM'
];
const urlSafeB64Pad = [
  'SGVsbG8gVGVuc29yRmxvdy5qcyE=', '8J2Mhg==',
  'UHJl4oCUdHJhaW5lZCBtb2RlbHMgd2l0aCBCYXNlNjQgb3BzLg==',
  'aG93IGFib3V0IHRoZXNlPyDwn4yN8J-Su_CfjZU=',
  'aHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvanM=', 'w6DOssOHZMOpZg==',
  '5L2g5aW9LCDkuJbnlYw=', 'QnVpbGQsIHRyYWluLCAmIGRlcGxveQpNTCBtb2RlbHMgaW4gSlM='
];

describeWithFlags('encodeBase64', ALL_ENVS, () => {
  it('scalar', async () => {
    const a = tf.scalar(txtArr[1], 'string');
    const r = tf.encodeBase64(a);
    expect(r.shape).toEqual([]);
    expectArraysEqual(await r.data(), urlSafeB64[1]);
  });
  it('1D padded', async () => {
    const a = tf.tensor1d([txtArr[2]], 'string');
    const r = tf.encodeBase64(a, true);
    expect(r.shape).toEqual([1]);
    expectArraysEqual(await r.data(), [urlSafeB64Pad[2]]);
  });
  it('2D', async () => {
    const a = tf.tensor2d(txtArr, [2, 4], 'string');
    const r = tf.encodeBase64(a, false);
    expect(r.shape).toEqual([2, 4]);
    expectArraysEqual(await r.data(), urlSafeB64);
  });
  it('3D padded', async () => {
    const a = tf.tensor3d(txtArr, [2, 2, 2], 'string');
    const r = tf.encodeBase64(a, true);
    expect(r.shape).toEqual([2, 2, 2]);
    expectArraysEqual(await r.data(), urlSafeB64Pad);
  });
});

describeWithFlags('decodeBase64', ALL_ENVS, () => {
  it('scalar', async () => {
    const a = tf.scalar(urlSafeB64[1], 'string');
    const r = tf.decodeBase64(a);
    expect(r.shape).toEqual([]);
    expectArraysEqual(await r.data(), txtArr[1]);
  });
  it('1D padded', async () => {
    const a = tf.tensor1d([urlSafeB64Pad[2]], 'string');
    const r = tf.decodeBase64(a);
    expect(r.shape).toEqual([1]);
    expectArraysEqual(await r.data(), [txtArr[2]]);
  });
  it('2D', async () => {
    const a = tf.tensor2d(urlSafeB64, [2, 4], 'string');
    const r = tf.decodeBase64(a);
    expect(r.shape).toEqual([2, 4]);
    expectArraysEqual(await r.data(), txtArr);
  });
  it('3D padded', async () => {
    const a = tf.tensor3d(urlSafeB64Pad, [2, 2, 2], 'string');
    const r = tf.decodeBase64(a);
    expect(r.shape).toEqual([2, 2, 2]);
    expectArraysEqual(await r.data(), txtArr);
  });
});

describeWithFlags('encodeBase64-decodeBase64', ALL_ENVS, () => {
  it('round-trip', async () => {
    const s = [txtArr.join('')];
    const a = tf.tensor(s, [1], 'string');
    const b = tf.encodeBase64(a);
    const c = tf.decodeBase64(b);
    expectArraysEqual(await c.data(), s);
  });
});
