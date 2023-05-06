/**
 * @license
 * Copyright 2023 Google LLC.
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
import {BROWSER_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysEqual} from '../test_util';

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

describeWithFlags('Draw on 2d context', BROWSER_ENVS, () => {
  it('draw image with 4 channels and int values', async () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const img = tf.tensor3d(data, [2, 2, 4], 'int32');
    const canvas = new MockCanvas(2, 2);
    const ctx = canvas.getContext('2d');

    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, {contextOptions: {contextType: '2d'}});
    expectArraysEqual(ctx.getImageData().data, data);
  });

  it('draw image with 4 channels and float values', async () => {
    const data =
        [.1, .2, .3, .4, .5, .6, .7, .8, .9, .1, .11, .12, .13, .14, .15, .16];
    const img = tf.tensor3d(data, [2, 2, 4]);
    const canvas = new MockCanvas(2, 2);
    const ctx = canvas.getContext('2d');

    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, {contextOptions: {contextType: '2d'}});
    const actualData = ctx.getImageData().data;
    const expectedData = data.map(e => Math.round(e * 255));
    expectArraysEqual(actualData, expectedData);
  });

  it('draw 2D image in grayscale', async () => {
    const data = [1, 2, 3, 4];
    const img = tf.tensor2d(data, [2, 2], 'int32');
    const canvas = new MockCanvas(2, 2);
    const ctx = canvas.getContext('2d');

    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, {contextOptions: {contextType: '2d'}});
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

    const drawOptions = {
      contextOptions: {contextType: '2d'},
      imageOptions: {alpha: 0.5}
    };
    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, drawOptions);
    const actualData = ctx.getImageData().data;
    const expectedData =
        [1, 1, 1, 128, 2, 2, 2, 128, 3, 3, 3, 128, 4, 4, 4, 128];
    expectArraysEqual(actualData, expectedData);
  });
});
