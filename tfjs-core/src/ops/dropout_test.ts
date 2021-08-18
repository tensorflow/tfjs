/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {expectArraysClose} from '../test_util';

import {tensor1d, tensor2d} from './ops';

describeWithFlags('dropout', ALL_ENVS, () => {
  it('x 1d array, rate 0', async () => {
    const x = tensor1d([1, 2, 2, 1]);
    const rate = 0;
    const output = tf.dropout(x, rate);
    expect(output.dtype).toEqual(x.dtype);
    expect(output.shape).toEqual(x.shape);
    expectArraysClose(await x.data(), await output.data());
  });

  it('x 1d array, rate 0.75', async () => {
    const x = tensor1d([1, 2, 2, 1]);
    const rate = 0.75;
    const output = tf.dropout(x, rate);
    expect(output.dtype).toEqual(x.dtype);
    expect(output.shape).toEqual(x.shape);
    const xValues = await x.data();
    const outputValues = await output.data();
    for (let i = 0; i < xValues.length; i++) {
      if (outputValues[i] !== 0) {
        expect(outputValues[i]).toBeCloseTo(1 / (1 - rate) * xValues[i]);
      }
    }
  });

  it('x 2d array, rate 0', async () => {
    const x = tensor2d([1, 5, 2, 4, 3, 6], [2, 3]);
    const rate = 0;
    const output = tf.dropout(x, rate);
    expect(output.dtype).toEqual(x.dtype);
    expect(output.shape).toEqual(x.shape);
    expectArraysClose(await x.data(), await output.data());
  });

  it('x 2d array, rate 0.75', async () => {
    const x = tensor2d([1, 5, 2, 4, 3, 6], [2, 3]);
    const rate = 0.75;
    const output = tf.dropout(x, rate);
    expect(output.dtype).toEqual(x.dtype);
    expect(output.shape).toEqual(x.shape);
    const xValues = await x.data();
    const outputValues = await output.data();
    for (let i = 0; i < xValues.length; i++) {
      if (outputValues[i] !== 0) {
        expect(outputValues[i]).toBeCloseTo(1 / (1 - rate) * xValues[i]);
      }
    }
  });

  it('x 1d array, rate 0.75, with noise shape length = 1', async () => {
    const x = tensor1d([1, 2, 2, 1]);
    const rate = 0.75;
    const noiseShape = [1];
    const output = tf.dropout(x, rate, noiseShape);
    expect(output.dtype).toEqual(x.dtype);
    expect(output.shape).toEqual(x.shape);
    const xValues = await x.data();
    const outputValues = await output.data();
    const maskedOutput = outputValues[0];
    for (let i = 0; i < xValues.length; i++) {
      if (maskedOutput === 0) {
        expect(outputValues[i]).toBe(maskedOutput);
      }
      if (outputValues[i] !== 0) {
        expect(outputValues[i]).toBeCloseTo(1 / (1 - rate) * xValues[i]);
      }
    }
  });

  it('x 2d array, rate 0.75, with noise shape length = 2', async () => {
    const x = tensor2d([1, 5, 2, 4, 3, 6], [2, 3]);
    const rate = 0.75;
    const noiseShape = [2, 1];
    const output = tf.dropout(x, rate, noiseShape);
    expect(output.dtype).toEqual(x.dtype);
    expect(output.shape).toEqual(x.shape);
    const xValues = await x.data();
    const outputValues = await output.data();
    for (let i = 0; i < x.shape[0]; i++) {
      const maskedOutput = outputValues[i * x.shape[1]];
      if (maskedOutput !== 0) {
        expect(maskedOutput)
            .toBeCloseTo(1 / (1 - rate) * xValues[i * x.shape[1]]);
      } else {
        for (let j = 0; j < x.shape[1]; j++) {
          expect(outputValues[i * x.shape[1] + j]).toBe(maskedOutput);
        }
      }
    }
  });

  it('broadcast noise shape', async () => {
    const x = tensor2d([1, 5, 2, 4, 3, 6], [2, 3]);
    const rate = 0.75;
    // broadcast noise shape, same output as using noiseShape [2, 1]
    const noiseShape = [1];
    const output = tf.dropout(x, rate, noiseShape);
    expect(output.dtype).toEqual(x.dtype);
    expect(output.shape).toEqual(x.shape);
    const xValues = await x.data();
    const outputValues = await output.data();
    for (let i = 0; i < x.shape[0]; i++) {
      const maskedOutput = outputValues[i * x.shape[1]];
      if (maskedOutput !== 0) {
        expect(maskedOutput)
            .toBeCloseTo(1 / (1 - rate) * xValues[i * x.shape[1]]);
      } else {
        for (let j = 0; j < x.shape[1]; j++) {
          expect(outputValues[i * x.shape[1] + j]).toBe(maskedOutput);
        }
      }
    }
  });

  it('x 1d array, rate 0.75, with seed', async () => {
    const x = tensor1d([1, 2, 2, 1]);
    const rate = 0.75;
    const seed = 23;
    const output = tf.dropout(x, rate, null, seed);
    expect(output.dtype).toEqual(x.dtype);
    expect(output.shape).toEqual(x.shape);
    const xValues = await x.data();
    const outputValues = await output.data();
    for (let i = 0; i < xValues.length; i++) {
      if (outputValues[i] !== 0) {
        expect(outputValues[i]).toBeCloseTo(1 / (1 - rate) * xValues[i]);
      }
    }
  });

  it('x TensorLike object', async () => {
    const x = [1.0, 2.0, 2.0, 1.0];
    const rate = 0;
    const output = tf.dropout(x, rate);
    expect(output.dtype).toEqual('float32');
    expect(output.shape).toEqual([4]);
    expectArraysClose(await output.data(), x);
  });

  it('throws when x.dtype != float32', async () => {
    const x = tensor1d([1, 2, 2, 1], 'int32');
    const rate = 0.75;
    expect(() => tf.dropout(x, rate)).toThrowError();
  });

  it('throws when rate is not in the range [0, 1)', async () => {
    const x = tensor1d([1, 2, 2, 1]);
    const rate = 1.5;
    expect(() => tf.dropout(x, rate)).toThrowError();
  });
});
