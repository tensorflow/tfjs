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

/**
 * Unit Tests for StartEndPacker Layer.
 */

import { Tensor, tensor } from '@tensorflow/tfjs-core';

import { StartEndPacker } from './start_end_packer';
import { expectTensorsClose } from '../../../utils/test_utils';
import { ValueError } from '../../../errors';

describe('StartEndPacker', () => {
  it('tensor input', () => {
    const inputData = tensor([5, 6, 7]);
    const startEndPacker = new StartEndPacker({sequenceLength: 5});

    const output = startEndPacker.call(inputData) as Tensor;
    const expectedOutput = tensor([5, 6, 7, 0, 0]);

    expectTensorsClose(output, expectedOutput);
  });

  it('tensor array input', () => {
    const inputData = [tensor([5, 6, 7])];
    const startEndPacker = new StartEndPacker({sequenceLength: 5});

    const output = startEndPacker.call(inputData) as Tensor[];
    const expectedOutput = [tensor([5, 6, 7, 0, 0])];

    expect(output.length).toBe(1);
    expectTensorsClose(output[0], expectedOutput[0]);
  });

  it('uneven tensor array input', () => {
    const inputData = [tensor([5, 6, 7]), tensor([8, 9, 10, 11])];
    const startEndPacker = new StartEndPacker({sequenceLength: 5});

    const output = startEndPacker.call(inputData) as Tensor[];
    const expectedOutput = [tensor([5, 6, 7, 0, 0]), tensor([8, 9, 10, 11, 0])];

    expect(output.length).toBe(2);
    expectTensorsClose(output[0], expectedOutput[0]);
    expectTensorsClose(output[1], expectedOutput[1]);
  });

  it('tensor array truncation', () => {
    const inputData = [tensor([5, 6, 7]), tensor([8, 9, 10, 11])];
    const startEndPacker = new StartEndPacker({sequenceLength: 3});

    const output = startEndPacker.call(inputData) as Tensor[];
    const expectedOutput = [tensor([5, 6, 7]), tensor([8, 9, 10])];

    expect(output.length).toBe(2);
    expectTensorsClose(output[0], expectedOutput[0]);
    expectTensorsClose(output[1], expectedOutput[1]);
  });

  it('tensor array input error', () => {
    const inputData = [tensor([[5, 6, 7], [8, 9, 10, 11]])];
    const startEndPacker = new StartEndPacker({sequenceLength: 5});

    expect(() => startEndPacker.call(inputData))
      .toThrow(new ValueError(
        'Input must either be a rank 1 Tensor or an array of rank 1 Tensors.'));
  });

  it('start end token', () => {
    const inputData = [tensor([5, 6, 7]), tensor([8, 9, 10, 11])];
    const startEndPacker = new StartEndPacker({
      sequenceLength: 6, startValue: 1, endValue: 2,
    });

    const output = startEndPacker.call(inputData) as Tensor[];
    const expectedOutput = [
      tensor([1, 5, 6, 7, 2, 0]), tensor([1, 8, 9, 10, 11, 2])
    ];

    expect(output.length).toBe(2);
    expectTensorsClose(output[0], expectedOutput[0]);
    expectTensorsClose(output[1], expectedOutput[1]);
  });

  it('start end and padding', () => {
    const inputData = [tensor([5, 6, 7]), tensor([8, 9, 10, 11])];
    const startEndPacker = new StartEndPacker({
      sequenceLength: 7, startValue: 1, endValue: 2, padValue: 3,
    });

    const output = startEndPacker.call(inputData) as Tensor[];
    const expectedOutput = [
      tensor([1, 5, 6, 7, 2, 3, 3]), tensor([1, 8, 9, 10, 11, 2, 3])
    ];

    expect(output.length).toBe(2);
    expectTensorsClose(output[0], expectedOutput[0]);
    expectTensorsClose(output[1], expectedOutput[1]);
  });

  it('end token value during truncation', () => {
    const inputData = [tensor([5, 6]), tensor([8, 9, 10, 11, 12, 13])];
    const startEndPacker = new StartEndPacker({
      sequenceLength: 5, startValue: 1, endValue: 2, padValue: 0,
    });

    const output = startEndPacker.call(inputData) as Tensor[];
    const expectedOutput = [tensor([1, 5, 6, 2, 0]), tensor([1, 8, 9, 10, 2])];

    expect(output.length).toBe(2);
    expectTensorsClose(output[0], expectedOutput[0]);
    expectTensorsClose(output[1], expectedOutput[1]);
  });

  it('string input', () => {
    const inputData = [
      tensor(['TensorflowJS', 'is', 'awesome']),
      tensor(['amazing'])
    ];
    const startEndPacker = new StartEndPacker({
      sequenceLength: 5,
      startValue: '[START]',
      endValue: '[END]',
      padValue: '[PAD]',
    });

    const output = startEndPacker.call(inputData) as Tensor[];
    const expectedOutput = [
      tensor(['[START]', 'TensorflowJS', 'is', 'awesome', '[END]']),
      tensor(['[START]', 'amazing', '[END]', '[PAD]', '[PAD]']),
    ];

    expect(output.length).toBe(2);
    expectTensorsClose(output[0], expectedOutput[0]);
    expectTensorsClose(output[1], expectedOutput[1]);
  });

  it('correct getConfig', () => {
    const startEndPacker = new StartEndPacker({
      sequenceLength: 512,
      startValue: 10,
      endValue: 20,
      padValue: 100,
    });
    const config = startEndPacker.getConfig();

    expect(config.sequenceLength).toEqual(512);
    expect(config.startValue).toEqual(10);
    expect(config.endValue).toEqual(20);
    expect(config.padValue).toEqual(100);
  });
});
