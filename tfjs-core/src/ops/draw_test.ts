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
import {expectArraysClose, expectArraysEqual} from '../test_util';

function readPixelsFromCanvas(
    canvas: HTMLCanvasElement, contextType: string, width: number,
    height: number) {
  let actualData;
  if (contextType === '2d') {
    const ctx = canvas.getContext(contextType);
    actualData = ctx.getImageData(0, 0, width, height).data;
  } else {
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = width;
    tmpCanvas.height = height;
    const ctx = tmpCanvas.getContext('2d');

    ctx.drawImage(canvas, 0, 0);
    actualData = new Uint8ClampedArray(
        ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height).data);
  }
  return actualData;
}

function convertToRGBA(
    data: number[], shape: number[], dtype: string, alpha = 1) {
  const [height, width] = shape.slice(0, 2);
  const depth = shape.length === 2 ? 1 : shape[2];
  const multiplier = dtype === 'float32' ? 255 : 1;
  const bytes = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < height * width; ++i) {
    const rgba = [0, 0, 0, 255 * alpha];

    for (let d = 0; d < depth; d++) {
      const value = data[i * depth + d];

      if (dtype === 'float32') {
        if (value < 0 || value > 1) {
          throw new Error(
              `Tensor values for a float32 Tensor must be in the ` +
              `range [0 - 1] but encountered ${value}.`);
        }
      } else if (dtype === 'int32') {
        if (value < 0 || value > 255) {
          throw new Error(
              `Tensor values for a int32 Tensor must be in the ` +
              `range [0 - 255] but encountered ${value}.`);
        }
      }

      if (depth === 1) {
        rgba[0] = value * multiplier;
        rgba[1] = value * multiplier;
        rgba[2] = value * multiplier;
      } else {
        rgba[d] = value * multiplier;
      }
    }

    const j = i * 4;
    bytes[j + 0] = Math.round(rgba[0]);
    bytes[j + 1] = Math.round(rgba[1]);
    bytes[j + 2] = Math.round(rgba[2]);
    bytes[j + 3] = Math.round(rgba[3]);
  }
  return bytes;
}

function drawAndReadback(
    contextType: string, data: number[], shape: number[], dtype: string,
    alpha = 1, canvasUsedAs2d = false) {
  const [height, width] = shape.slice(0, 2);
  let img;
  if (shape.length === 3) {
    img = tf.tensor3d(
        data, shape as [number, number, number], dtype as keyof tf.DataTypeMap);
  } else {
    img = tf.tensor2d(
        data, shape as [number, number], dtype as keyof tf.DataTypeMap);
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  if (canvasUsedAs2d) {
    canvas.getContext('2d');
  }
  const drawOptions = {contextOptions: {contextType}, imageOptions: {alpha}};
  // tslint:disable-next-line:no-any
  tf.browser.draw(img, canvas as any, drawOptions);
  const actualData = readPixelsFromCanvas(canvas, contextType, width, height);
  const expectedData = convertToRGBA(data, shape, dtype, alpha);
  img.dispose();
  return [actualData, expectedData];
}

// CPU and GPU handle pixel value differently. The epsilon may possibly grow
// after each draw and read back.
const DRAW_EPSILON = 6.0;

describeWithFlags('draw on canvas context', BROWSER_ENVS, (env) => {
  let contextType: string;
  beforeAll(async () => {
    await tf.setBackend(env.name);
    contextType = env.name === 'cpu' ? '2d' : env.name;
  });

  it('draw image with 4 channels and int values', async () => {
    const data =
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
    const shape = [2, 2, 4];
    const dtype = 'int32';
    const startNumTensors = tf.memory().numTensors;
    const [actualData, expectedData] =
        drawAndReadback(contextType, data, shape, dtype);
    expect(tf.memory().numTensors).toEqual(startNumTensors);
    expectArraysClose(actualData, expectedData, DRAW_EPSILON);
  });

  it('draw image with 4 channels and int values, alpha=0.5', async () => {
    const data =
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
    const shape = [2, 2, 4];
    const dtype = 'int32';
    const startNumTensors = tf.memory().numTensors;
    const [actualData, expectedData] =
        drawAndReadback(contextType, data, shape, dtype, 0.5);
    expect(tf.memory().numTensors).toEqual(startNumTensors);
    expectArraysClose(actualData, expectedData, DRAW_EPSILON);
  });

  it('draw image with 4 channels and float values', async () => {
    const data =
        [.1, .2, .3, .4, .5, .6, .7, .8, .09, .1, .11, .12, .13, .14, .15, .16];
    const shape = [2, 2, 4];
    const dtype = 'float32';
    const [actualData, expectedData] =
        drawAndReadback(contextType, data, shape, dtype);
    expectArraysClose(actualData, expectedData, DRAW_EPSILON);
  });

  it('draw image with 3 channels and int values', async () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const shape = [2, 2, 3];
    const dtype = 'int32';
    const [actualData, expectedData] =
        drawAndReadback(contextType, data, shape, dtype);
    expectArraysEqual(actualData, expectedData);
  });

  it('draw image with 3 channels and int values, alpha=0.5', async () => {
    const data = [101, 32, 113, 14, 35, 76, 17, 38, 59, 70, 81, 92];
    const shape = [2, 2, 3];
    const dtype = 'int32';
    const alpha = 0.5;
    const [actualData, expectedData] =
        drawAndReadback(contextType, data, shape, dtype, alpha);
    expectArraysClose(actualData, expectedData, DRAW_EPSILON);
  });

  it('draw image with 3 channels and float values', async () => {
    const data = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .1, .11, .12];
    const shape = [2, 2, 3];
    const dtype = 'float32';
    const [actualData, expectedData] =
        drawAndReadback(contextType, data, shape, dtype);
    expectArraysClose(actualData, expectedData, DRAW_EPSILON);
  });

  it('draw 2D image in grayscale', async () => {
    const data = [100, 12, 90, 64];
    const shape = [2, 2];
    const dtype = 'int32';
    const [actualData, expectedData] =
        drawAndReadback(contextType, data, shape, dtype);
    expectArraysEqual(actualData, expectedData);
  });

  it('draw image with alpha=0.5', async () => {
    const data = [101, 212, 113, 14, 35, 76, 17, 38, 59, 70, 81, 92];
    const shape = [6, 2, 1];
    const dtype = 'int32';
    const alpha = 0.5;
    const [actualData, expectedData] =
        drawAndReadback(contextType, data, shape, dtype, alpha);
    expectArraysClose(actualData, expectedData, DRAW_EPSILON);
  });

  it('draw image works when canvas has been used for 2d', async () => {
    const data = [101, 212, 113, 14, 35, 76, 17, 38, 59, 70, 81, 92];
    const shape = [6, 2, 1];
    const dtype = 'int32';
    // Set canvasUsedAs2d to true so the canvas will be first used for 2d.
    const [actualData, expectedData] =
        drawAndReadback(contextType, data, shape, dtype, 1, true);
    expectArraysClose(actualData, expectedData, DRAW_EPSILON);
  });
});
