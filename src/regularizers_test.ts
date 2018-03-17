/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Unit tests for constraints */

import {tensor1d} from '@tensorflow/tfjs-core';
import * as tfl from './index';

// tslint:disable:max-line-length
import {deserializeRegularizer, getRegularizer, serializeRegularizer} from './regularizers';
import {ConfigDict, LayerVariable} from './types';
import {describeMathCPU, expectTensorsClose} from './utils/test_utils';
// tslint:enable:max-line-length

describeMathCPU('Built-in Regularizers', () => {
  it('l1_l2', () => {
    const variable = new LayerVariable(tensor1d([1, -2, 3, -4]));
    const regularizer = tfl.regularizers.l1l2();
    const score = regularizer.apply(variable);
    expectTensorsClose(
        score, tensor1d([0.01 * (1 + 2 + 3 + 4) + 0.01 * (1 + 4 + 9 + 16)]));
  });
  it('l1', () => {
    const variable = new LayerVariable(tensor1d([1, -2, 3, -4]));
    const regularizer = tfl.regularizers.l1();
    const score = regularizer.apply(variable);
    expectTensorsClose(score, tensor1d([0.01 * (1 + 2 + 3 + 4)]));
  });
  it('l2', () => {
    const variable = new LayerVariable(tensor1d([1, -2, 3, -4]));
    const regularizer = tfl.regularizers.l2();
    const score = regularizer.apply(variable);
    expectTensorsClose(score, tensor1d([0.01 * (1 + 4 + 9 + 16)]));
  });
  it('l1_l2 non default', () => {
    const variable = new LayerVariable(tensor1d([1, -2, 3, -4]));
    const regularizer = tfl.regularizers.l1l2({l1: 1, l2: 2});
    const score = regularizer.apply(variable);
    expectTensorsClose(
        score, tensor1d([1 * (1 + 2 + 3 + 4) + 2 * (1 + 4 + 9 + 16)]));
  });
});

describeMathCPU('regularizers.get', () => {
  let variable: LayerVariable;
  beforeEach(() => {
    variable = new LayerVariable(tensor1d([1, -2, 3, -4]));
  });

  it('by string', () => {
    const regularizer = getRegularizer('L1L2');
    expectTensorsClose(
        regularizer.apply(variable), tfl.regularizers.l1l2().apply(variable));
  });
  it('by existing object', () => {
    const origReg = tfl.regularizers.l1l2({l1: 1, l2: 2});
    const regularizer = getRegularizer(origReg);
    expect(regularizer).toEqual(origReg);
  });
  it('by config dict', () => {
    const origReg = tfl.regularizers.l1l2({l1: 1, l2: 2});
    const regularizer =
        getRegularizer(serializeRegularizer(origReg) as ConfigDict);
    expectTensorsClose(regularizer.apply(variable), origReg.apply(variable));
  });
});

describeMathCPU('Regularizer Serialization', () => {
  it('Built-ins', () => {
    const regularizer = tfl.regularizers.l1l2({l1: 1, l2: 2});
    const config = serializeRegularizer(regularizer) as ConfigDict;
    const reconstituted = deserializeRegularizer(config);
    const roundTripConfig = serializeRegularizer(reconstituted) as ConfigDict;
    expect(roundTripConfig.className).toEqual('L1L2');
    const nestedConfig = roundTripConfig.config as ConfigDict;
    expect(nestedConfig.l1).toEqual(1);
    expect(nestedConfig.l2).toEqual(2);
  });
});
