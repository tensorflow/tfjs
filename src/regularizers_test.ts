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

import {tensor1d} from 'deeplearn';

import {deserialize, get, l1, l1_l2, L1L2, l2, serialize} from './regularizers';
import {ConfigDict, LayerVariable} from './types';
import {describeMathCPU, expectTensorsClose} from './utils/test_utils';

describeMathCPU('Built-in Regularizers', () => {
  it('l1_l2', () => {
    const variable = new LayerVariable(tensor1d([1, -2, 3, -4]));
    const regularizer = l1_l2();
    const score = regularizer.apply(variable);
    expectTensorsClose(
        score, tensor1d([0.01 * (1 + 2 + 3 + 4) + 0.01 * (1 + 4 + 9 + 16)]));
  });
  it('l1', () => {
    const variable = new LayerVariable(tensor1d([1, -2, 3, -4]));
    const regularizer = l1();
    const score = regularizer.apply(variable);
    expectTensorsClose(score, tensor1d([0.01 * (1 + 2 + 3 + 4)]));
  });
  it('l2', () => {
    const variable = new LayerVariable(tensor1d([1, -2, 3, -4]));
    const regularizer = l2();
    const score = regularizer.apply(variable);
    expectTensorsClose(score, tensor1d([0.01 * (1 + 4 + 9 + 16)]));
  });
  it('l1_l2 non default', () => {
    const variable = new LayerVariable(tensor1d([1, -2, 3, -4]));
    const regularizer = l1_l2(1, 2);
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
    const regularizer = get('L1L2');
    expectTensorsClose(
        regularizer.apply(variable), (new L1L2()).apply(variable));
  });
  it('by existing object', () => {
    const origReg = l1_l2(1, 2);
    const regularizer = get(origReg);
    expect(regularizer).toEqual(origReg);
  });
  it('by config dict', () => {
    const origReg = l1_l2(1, 2);
    const regularizer = get(serialize(origReg) as ConfigDict);
    expectTensorsClose(regularizer.apply(variable), origReg.apply(variable));
  });
});

describeMathCPU('Regularizer Serialization', () => {
  it('Built-ins', () => {
    const regularizer = l1_l2(1, 2);
    const config = serialize(regularizer) as ConfigDict;
    const reconstituted = deserialize(config);
    const roundTripConfig = serialize(reconstituted) as ConfigDict;
    expect(roundTripConfig.className).toEqual('L1L2');
    const nestedConfig = roundTripConfig.config as ConfigDict;
    expect(nestedConfig.l1).toEqual(1);
    expect(nestedConfig.l2).toEqual(2);
  });
});
