/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { describeMathCPUAndGPU, expectTensorsClose} from '../../utils/test_utils';
import { Tensor, tensor, tensor1d} from '@tensorflow/tfjs-core';
import * as utils from './preprocessing_utils';

describeMathCPUAndGPU('Preprocessing Utils', () => {

  it('Peforms int encoding correctly', () => {
    const inputs = tensor([0, 1, 2], [3], 'int32');
    const outputs = utils.
    encodeCategoricalInputs(inputs, 'int', 4) as Tensor;
    expectTensorsClose(outputs, inputs);
  });

  it('Peforms oneHot encoding correctly', () => {
    const inputs = tensor([0, 1, 2],  [3], 'int32');
    const outputs = utils.
    encodeCategoricalInputs(inputs, 'oneHot', 4) as Tensor;
    const expectedOutput = tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]);
    expectTensorsClose(outputs, expectedOutput);
  });

  it('Peforms multiHot encoding correctly', () => {
    const inputs = tensor([0, 1, 2], [3], 'int32');
    const outputs = utils.
    encodeCategoricalInputs(inputs, 'multiHot', 4) as Tensor;
    const expectedOutput = tensor([1, 1, 1, 0]);
    expectTensorsClose(outputs, expectedOutput);
  });

  it('Peforms count encoding correctly', () => {
    const inputs = tensor([0, 1, 1, 2, 2, 2], [6], 'int32');
    const outputs = utils.
    encodeCategoricalInputs(inputs, 'count', 4) as Tensor;
    const expectedOutput = tensor([1, 2, 3, 0]);
    expectTensorsClose(outputs, expectedOutput);
  });

  it('Peforms tfIdf encoding correctly', () => {
    const inputs = tensor([0, 1, 1, 2, 2, 2]);
    const idfWeights = tensor1d([0.1, 1.0, 10.0, 0]);
    const outputs = utils.encodeCategoricalInputs(inputs, 'tfIdf',
                                                  4, idfWeights) as Tensor;
    const expectedOutput = tensor([0.1, 2, 30, 0]);
    expectTensorsClose(outputs, expectedOutput);
  });

  it('Thows an error if input rank > 2', () => {
    const inputs = tensor([[[1], [2]], [[3], [1]]]);
    expect(() =>utils.encodeCategoricalInputs(inputs, 'multiHot', 4))
    .toThrowError(`When outputMode is not int, maximum output rank is 2`
    + ` Received outputMode ${'multiHot'} and input shape ${inputs.shape}`
    + ` which would result in output rank ${inputs.rank}.`);
  });

  it('Thows an error if weights are not supplied for tfIdf', () => {
    const inputs = tensor([0, 1, 1, 2, 2, 2]);
    expect(() =>utils.encodeCategoricalInputs(inputs, 'tfIdf', 4)).
    toThrowError(`When outputMode is 'tfIdf', weights must be provided.`);
  });

});
