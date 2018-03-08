/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit tests for addition_rnn.ts.
 */

// tslint:disable:max-line-length
import {tensor2d} from 'deeplearn';
import * as tfl from 'tfjs-layers';

import {CharacterTable, convertDataToTensors, createAndCompileModel, generateData} from './addition_rnn';

// tslint:enable:max-line-length

tfl.describeMathCPU('CharacterTable', () => {
  it('encode success', () => {
    const table = new CharacterTable('01234 ');
    tfl.expectTensorsClose(table.encode('12 ', 4), tensor2d([
                             [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]
                           ]));
  });
  it('encode out of vocabulary', () => {
    const table = new CharacterTable('01234 ');
    expect(() => table.encode('5', 1)).toThrowError(/Unknown character/);
  });
  it('decode success', () => {
    const table = new CharacterTable('01234 ');
    const decoded = table.decode(
        tensor2d([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]]));
    expect(decoded).toEqual('12 ');
  });
});

tfl.describeMathCPU('generateData', () => {
  it('no invert', () => {
    const data = generateData(2, 8, false);
    expect(data.length).toEqual(8);
    for (const [question, expected] of data) {
      expect(question.length).toEqual(5);
      const trimmedQuestion = question.trim();
      const indexPlus = trimmedQuestion.indexOf('+');
      const aStr = trimmedQuestion.slice(0, indexPlus);
      expect(aStr.length).toBeLessThanOrEqual(2);
      const bStr = trimmedQuestion.slice(indexPlus + 1);
      expect(bStr.length).toBeLessThanOrEqual(2);
      const a = Number.parseInt(aStr);
      const b = Number.parseInt(bStr);
      expect(expected.length).toBeLessThanOrEqual(3);
      expect(a + b).toEqual(Number.parseInt(expected));
    }
  });
  it('convertDataToTensors', () => {
    const digits = 2;
    const maxLen = digits + 1 + digits;
    const numExamples = 8;
    const data = generateData(digits, numExamples, false);
    const chars = '0123456789+ ';
    const charTable = new CharacterTable(chars);
    const [x, y] = convertDataToTensors(data, charTable, digits);
    expect(x.shape).toEqual([numExamples, maxLen, chars.length]);
    expect(y.shape).toEqual([numExamples, digits + 1, chars.length]);
  });
});

tfl.describeMathCPUAndGPU('AdditionRNN', () => {
  it('train with data', async done => {
    const digits = 2;
    const numExamples = 10;
    const chars = '0123456789+ ';
    const charTable = new CharacterTable(chars);
    const model =
        createAndCompileModel(1, 32, 'SimpleRNN', digits, chars.length);

    const trainData = generateData(digits, numExamples, false);
    const [trainXs, trainYs] =
        convertDataToTensors(trainData, charTable, digits);
    const testData = generateData(digits, numExamples, false);
    const [testXs, testYs] = convertDataToTensors(testData, charTable, digits);
    await model.fit({
      x: trainXs,
      y: trainYs,
      epochs: 3,
      batchSize: 4,
      validationData: [testXs, testYs]
    });
    done();
  });
});
