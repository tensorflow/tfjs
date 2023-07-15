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

import { Tensor1D, Tensor2D, memory, tensor, tensor2d } from '@tensorflow/tfjs-core';

import { StartEndPacker } from './start_end_packer';
import { expectTensorsClose } from '../../../utils/test_utils';
import { ValueError } from '../../../errors';

describe('StartEndPacker', () => {
  it('tensor input', () => {
    const inputData = tensor([5, 6, 7]);
    const startEndPacker = new StartEndPacker({sequenceLength: 5});

    const output = startEndPacker.call(inputData) as Tensor1D;
    const expectedOutput = tensor([5, 6, 7, 0, 0]);

    expectTensorsClose(output, expectedOutput);
  });

  it('tensor array input', () => {
    const inputData = [tensor([5, 6, 7])];
    const startEndPacker = new StartEndPacker({sequenceLength: 5});

    const output = startEndPacker.call(inputData) as Tensor2D;
    const expectedOutput = tensor2d([[5, 6, 7, 0, 0]]);

    expectTensorsClose(output, expectedOutput);
  });

  it('uneven tensor array input', () => {
    const inputData = [tensor([5, 6, 7]), tensor([8, 9, 10, 11])];
    const startEndPacker = new StartEndPacker({sequenceLength: 5});

    const output = startEndPacker.call(inputData) as Tensor2D;
    const expectedOutput = tensor2d([[5, 6, 7, 0, 0], [8, 9, 10, 11, 0]]);

    expectTensorsClose(output, expectedOutput);
  });

  it('tensor array truncation', () => {
    const inputData = [tensor([5, 6, 7]), tensor([8, 9, 10, 11])];
    const startEndPacker = new StartEndPacker({sequenceLength: 3});

    const output = startEndPacker.call(inputData) as Tensor2D;
    const expectedOutput = tensor2d([[5, 6, 7], [8, 9, 10]]);

    expectTensorsClose(output, expectedOutput);
  });

  it('tensor array input error', () => {
    const inputData = [tensor([[5, 6, 7], [8, 9, 10, 11]])];
    const startEndPacker = new StartEndPacker({sequenceLength: 5});

    expect(() => startEndPacker.call(inputData))
      .toThrow(new ValueError(
        'Input must either be a rank 1 Tensor or an array of rank 1 Tensors.'));
  });

  it('start only', () => {
    const inputData = [tensor([5, 6, 7]), tensor([8, 9, 10, 11])];
    const startEndPacker = new StartEndPacker({
      sequenceLength: 3, startValue: -1
    });

    const output = startEndPacker.call(inputData) as Tensor2D;
    const expectedOutput = tensor2d([[-1, 5, 6], [-1, 8, 9]]);

    expectTensorsClose(output, expectedOutput);
  });

  it('start end token', () => {
    const inputData = [tensor([5, 6, 7]), tensor([8, 9, 10, 11])];
    const startEndPacker = new StartEndPacker({
      sequenceLength: 6, startValue: 1, endValue: 2,
    });

    const output = startEndPacker.call(inputData) as Tensor2D;
    const expectedOutput = tensor2d([[1, 5, 6, 7, 2, 0], [1, 8, 9, 10, 11, 2]]);

    expectTensorsClose(output, expectedOutput);
  });

  it('start end and padding', () => {
    const inputData = [tensor([5, 6, 7]), tensor([8, 9, 10, 11])];
    const startEndPacker = new StartEndPacker({
      sequenceLength: 7, startValue: 1, endValue: 2, padValue: 3,
    });

    const output = startEndPacker.call(inputData) as Tensor2D;
    const expectedOutput = tensor2d([
      [1, 5, 6, 7, 2, 3, 3],
      [1, 8, 9, 10, 11, 2, 3],
    ]);

    expectTensorsClose(output, expectedOutput);
  });

  it('end token value during truncation', () => {
    const inputData = [tensor([5, 6]), tensor([8, 9, 10, 11, 12, 13])];
    const startEndPacker = new StartEndPacker({
      sequenceLength: 5, startValue: 1, endValue: 2, padValue: 0,
    });

    const output = startEndPacker.call(inputData) as Tensor2D;
    const expectedOutput = tensor2d([[1, 5, 6, 2, 0], [1, 8, 9, 10, 2]]);

    expectTensorsClose(output, expectedOutput);
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

    const output = startEndPacker.call(inputData) as Tensor2D;
    const expectedOutput = tensor2d([
      ['[START]', 'TensorflowJS', 'is', 'awesome', '[END]'],
      ['[START]', 'amazing', '[END]', '[PAD]', '[PAD]'],
    ]);

    expectTensorsClose(output, expectedOutput);
  });

  it('correct mask', () => {
    const inputData = [tensor([5, 6]), tensor([8, 9, 10, 11])];
    const startEndPacker = new StartEndPacker({
      sequenceLength: 6, startValue: 1, endValue: 2
    });

    const outputMask =
      startEndPacker.callAndReturnPaddingMask(inputData)[1] as Tensor2D;
    const expectedMask = tensor2d([
      [true, true, true, true, false, false],
      [true, true, true, true, true, true],
    ]);

    expectTensorsClose(outputMask, expectedMask);
  });

  it('does not leak memory', () => {
    const inputData = [tensor([5, 6]), tensor([8, 9, 10, 11])];
    const startEndPacker = new StartEndPacker({
      sequenceLength: 6, startValue: 1, endValue: 2
    });

    const numTensorsBefore = memory().numTensors;
    startEndPacker.callAndReturnPaddingMask(inputData);
    const numTensorsAfter = memory().numTensors;

    expect(numTensorsAfter).toEqual(numTensorsBefore + 2);
  });

  it('correct getConfig', () => {
    const config = {
      sequenceLength: 512,
      startValue: 10,
      endValue: 20,
      padValue: 100,
    };
    const startEndPacker = new StartEndPacker(config);
    const outputConfig = startEndPacker.getConfig();

    expect(outputConfig).toEqual(jasmine.objectContaining(config));
  });
});
