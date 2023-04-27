/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
import {test_util} from '@tensorflow/tfjs-core';

const {expectArraysEqual} = test_util;
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags, NODE_ENVS} from '@tensorflow/tfjs-core/dist/jasmine_util';


class MockContext {
  data: ImageData;

  getImageData() {
    return this.data;
  }

  putImageData(data: ImageData, x: number, y: number) {
    this.data = data;
  }
}

class MockCanvas {
  context: MockContext;

  constructor(public width: number, public height: number) {}

  getContext(type: '2d'): MockContext {
    if (this.context == null) {
      this.context = new MockContext();
    }
    return this.context;
  }
}

describeWithFlags('Draw', NODE_ENVS, () => {
  fit('draw image with 4 channels and int values', async () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const img = tf.tensor3d(data, [2, 2, 4], 'int32');
    const canvas = new MockCanvas(2, 2);
    const ctx = canvas.getContext('2d');

    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, {contextType: '2d'});
    expectArraysEqual(ctx.getImageData().data, data);
  });

  it('draw image with 4 channels and float values', async () => {
    const data =
        [.1, .2, .3, .4, .5, .6, .7, .8, .9, .1, .11, .12, .13, .14, .15, .16];
    const img = tf.tensor3d(data, [2, 2, 4]);
    const canvas = new MockCanvas(2, 2);
    const ctx = canvas.getContext('2d');

    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, {contextType: '2d'});
    const actualData = ctx.getImageData().data;
    const expectedData = data.map(e => Math.floor(e * 255));
    expectArraysEqual(actualData, expectedData);
  });

  it('draw 2D image as gray scale', async () => {
    const data = [1, 2, 3, 4];
    const img = tf.tensor2d(data, [2, 2], 'int32');
    const canvas = new MockCanvas(2, 2);
    const ctx = canvas.getContext('2d');

    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, {contextType: '2d'});
    const actualData = ctx.getImageData().data;
    const expectedData =
        [1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255, 4, 4, 4, 255];
    expectArraysEqual(actualData, expectedData);
  });

  it('draw image with alpha=0.5', async () => {
    const data = [1, 2, 3, 4];
    const img = tf.tensor3d(data, [2, 2, 1], 'int32');
    const canvas = new MockCanvas(2, 2);
    const ctx = canvas.getContext('2d');

    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, {contextType: '2d'}, {alpha: 0.5});
    const actualData = ctx.getImageData().data;
    const expectedData = [1, 1, 1, 63, 2, 2, 2, 63, 3, 3, 3, 63, 4, 4, 4, 63];
    expectArraysEqual(actualData, expectedData);
  });
});
