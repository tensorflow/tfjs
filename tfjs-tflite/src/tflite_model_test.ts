/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {DataType, NamedTensorMap} from '@tensorflow/tfjs-core';

import {TFLiteModel} from './tflite_model';
import {ProfileItem, TFLiteDataType, TFLiteWebModelRunner, TFLiteWebModelRunnerOptions, TFLiteWebModelRunnerTensorInfo} from './types/tflite_web_model_runner';

// A mock TFLiteWebModelRunner that doubles the data from input tensors to
// output tensors during inference.
class MockModelRunner implements TFLiteWebModelRunner {
  private mockInferResults: string[] = [];

  private inputTensors: TFLiteWebModelRunnerTensorInfo[];
  private outputTensors: TFLiteWebModelRunnerTensorInfo[];

  singleInput = false;
  singleOutput = false;

  constructor(
      modelPath: string, options: TFLiteWebModelRunnerOptions,
      firstOutputtype: TFLiteDataType = 'int32') {
    this.inputTensors = this.getTensorInfos();
    this.outputTensors = this.getTensorInfos(firstOutputtype);

    this.mockInferResults.push(`ModelPath=${modelPath}`);
    this.mockInferResults.push(`numThreads=${options.numThreads}`);
  }

  getInputs(): TFLiteWebModelRunnerTensorInfo[] {
    return this.singleInput ? [this.inputTensors[0]] : this.inputTensors;
  }

  getOutputs(): TFLiteWebModelRunnerTensorInfo[] {
    return this.singleOutput ? [this.outputTensors[0]] : this.outputTensors;
  }

  infer(): boolean {
    for (let i = 0; i < this.inputTensors.length; i++) {
      const inputTensor = this.inputTensors[i];
      const outputTensor = this.outputTensors[i];
      outputTensor.data().set(Array.from(inputTensor.data()).map(v => v * 2));
    }
    return true;
  }

  cleanUp() {}

  getProfilingResults(): ProfileItem[] {
    return [];
  }

  getProfilingSummary(): string {
    return '';
  }

  private getTensorInfos(firstTensorType: TFLiteDataType = 'int32'):
      TFLiteWebModelRunnerTensorInfo[] {
    const shape0 = [1, 2, 3];
    let buffer0: Int8Array|Uint8Array|Int16Array|Int32Array|Uint32Array|
        Float32Array|Float64Array = undefined;
    const size0 = shape0.reduce((a, c) => a * c, 1);
    switch (firstTensorType) {
      case 'int8':
        buffer0 = new Int8Array(size0);
        break;
      case 'uint8':
        buffer0 = new Uint8Array(size0);
        break;
      case 'int16':
        buffer0 = new Int16Array(size0);
        break;
      case 'int32':
        buffer0 = new Int32Array(size0);
        break;
      case 'uint32':
        buffer0 = new Uint32Array(size0);
        break;
      case 'float32':
        buffer0 = new Float32Array(size0);
        break;
      case 'float64':
        buffer0 = new Float64Array(size0);
        break;
      default:
        break;
    }

    const shape1 = [1, 2];
    const buffer1 = new Float32Array(shape1.reduce((a, c) => a * c, 1));
    return [
      {
        id: 0,
        dataType: firstTensorType,
        name: 't0',
        shape: shape0.join(','),
        data: () => buffer0,
      },
      {
        id: 1,
        dataType: 'float32',
        name: 't1',
        shape: shape1.join(','),
        data: () => buffer1,
      },
    ];
  }
}

let tfliteModel: TFLiteModel;
let modelRunner: MockModelRunner;

function checkOutputTypeConversion(
    originalOutputType: TFLiteDataType, expectedConvertedType: DataType) {
  const modelRunner = new MockModelRunner(
      'my_model.tflite', {numThreads: 2}, originalOutputType);
  modelRunner.singleInput = true;
  modelRunner.singleOutput = true;
  const tfliteModel = new TFLiteModel(modelRunner);

  const input = tf.tensor3d([1, 2, 3, 4, 5, 6], [1, 2, 3], 'int32');
  const outputs = tfliteModel.predict(input, {});
  expect((outputs as tf.Tensor).dtype).toBe(expectedConvertedType);
}

describe('TFLiteModel', () => {
  beforeEach(() => {
    modelRunner = new MockModelRunner('my_model.tflite', {numThreads: 2});
    tfliteModel = new TFLiteModel(modelRunner);
  });

  it('should generate the output for single tensor', () => {
    modelRunner.singleInput = true;

    const input = tf.tensor3d([1, 2, 3, 4, 5, 6], [1, 2, 3], 'int32');
    const outputs = tfliteModel.predict(input, {}) as NamedTensorMap;
    tf.test_util.expectArraysClose(
        outputs['t0'].dataSync(), [2, 4, 6, 8, 10, 12]);
  });

  it('should generate the output for tensor array', () => {
    const input0 = tf.tensor3d([1, 2, 3, 4, 5, 6], [1, 2, 3], 'int32');
    const input1 = tf.tensor2d([11, 12], [1, 2], 'float32');
    const outputs = tfliteModel.predict([input0, input1], {}) as NamedTensorMap;
    tf.test_util.expectArraysClose(
        outputs['t0'].dataSync(), [2, 4, 6, 8, 10, 12]);
    tf.test_util.expectArraysClose(outputs['t1'].dataSync(), [22, 24]);
  });

  it('should generate the output for tensor map', () => {
    const input0 = tf.tensor3d([1, 2, 3, 4, 5, 6], [1, 2, 3], 'int32');
    const input1 = tf.tensor2d([11, 12], [1, 2], 'float32');
    const outputs =
        tfliteModel.predict({'t0': input0, 't1': input1}, {}) as NamedTensorMap;
    tf.test_util.expectArraysClose(
        outputs['t0'].dataSync(), [2, 4, 6, 8, 10, 12]);
    tf.test_util.expectArraysClose(outputs['t1'].dataSync(), [22, 24]);
  });

  it('should generate a single output when model has a single output', () => {
    modelRunner.singleInput = true;
    modelRunner.singleOutput = true;

    const input = tf.tensor3d([1, 2, 3, 4, 5, 6], [1, 2, 3], 'int32');
    const outputs = tfliteModel.predict(input, {});
    expect(outputs instanceof tf.Tensor).toBe(true);
  });

  it('should convert output type correctly', () => {
    const testCases: Array<{
      originalOutputType: TFLiteDataType; expectedConvertedType: DataType;
    }> =
        [
          {
            originalOutputType: 'int8',
            expectedConvertedType: 'int32',
          },
          {
            originalOutputType: 'int16',
            expectedConvertedType: 'int32',
          },
          {
            originalOutputType: 'uint32',
            expectedConvertedType: 'int32',
          },
          {
            originalOutputType: 'float64',
            expectedConvertedType: 'float32',
          },
        ];

    for (const testCase of testCases) {
      checkOutputTypeConversion(
          testCase.originalOutputType, testCase.expectedConvertedType);
    }
  });

  it('should throw error if input size mismatch', () => {
    // Mismatch: 1 vs 2.
    const input0 = tf.tensor3d([1, 2, 3, 4, 5, 6], [1, 2, 3], 'int32');
    expect(() => tfliteModel.predict([input0], {})).toThrow();
  });

  it('should throw error if input shape mismatch', () => {
    // Mismatch: [2,2] vs [1,2,3].
    const input0 = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
    const input1 = tf.tensor2d([11, 12], [1, 2], 'float32');
    expect(() => tfliteModel.predict([input0, input1], {})).toThrow();
  });

  it('should throw error if input type is not compatible', () => {
    // Mismatch: float32 -> int32
    const input0 = tf.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
    const input1 = tf.tensor2d([11, 12], [1, 2], 'float32');
    expect(() => tfliteModel.predict([input0, input1], {})).toThrow();
  });
});
