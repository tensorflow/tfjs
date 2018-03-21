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

// tslint:disable:max-line-length
import {Tensor1D, tensor1d} from '@tensorflow/tfjs-core';

import {ConstraintIdentifier, deserializeConstraint, getConstraint, serializeConstraint} from './constraints';
import * as tfl from './index';
import {ConfigDict} from './types';
import {describeMathCPU, expectTensorsClose} from './utils/test_utils';

// tslint:enable:max-line-length

describeMathCPU('Built-in Constraints', () => {
  let initVals: Tensor1D;
  beforeEach(() => {
    initVals = tensor1d(new Float32Array([-1, 2, 0, 4, -5, 6]));
  });

  it('NonNeg', () => {
    const constraint = getConstraint('NonNeg');
    const postConstraint = constraint.apply(initVals);
    expectTensorsClose(
        postConstraint, tensor1d(new Float32Array([0, 2, 0, 4, 0, 6])));
  });

  it('MaxNorm', () => {
    const constraint = getConstraint('MaxNorm');
    const postConstraint = constraint.apply(initVals);
    expectTensorsClose(postConstraint, tensor1d(new Float32Array([
                         -0.2208630521, 0.4417261043, 0, 0.8834522086,
                         -1.104315261, 1.325178313
                       ])));
  });
  it('UnitNorm', () => {
    const constraint = getConstraint('UnitNorm');
    const postConstraint = constraint.apply(initVals);
    expectTensorsClose(postConstraint, tensor1d(new Float32Array([
                         -0.2208630521 / 2, 0.4417261043 / 2, 0,
                         0.8834522086 / 2, -1.104315261 / 2, 1.325178313 / 2
                       ])));
  });
  it('MinMaxNorm', () => {
    const constraint = getConstraint('MinMaxNorm');
    const postConstraint = constraint.apply(initVals);
    expectTensorsClose(postConstraint, tensor1d(new Float32Array([
                         -0.2208630521 / 2, 0.4417261043 / 2, 0,
                         0.8834522086 / 2, -1.104315261 / 2, 1.325178313 / 2
                       ])));
  });

  // Lower camel case.
  it('nonNeg', () => {
    const constraint = getConstraint('nonNeg');
    const postConstraint = constraint.apply(initVals);
    expectTensorsClose(
        postConstraint, tensor1d(new Float32Array([0, 2, 0, 4, 0, 6])));
  });

  it('maxNorm', () => {
    const constraint = getConstraint('maxNorm');
    const postConstraint = constraint.apply(initVals);
    expectTensorsClose(postConstraint, tensor1d(new Float32Array([
                         -0.2208630521, 0.4417261043, 0, 0.8834522086,
                         -1.104315261, 1.325178313
                       ])));
  });
  it('unitNorm', () => {
    const constraint = getConstraint('unitNorm');
    const postConstraint = constraint.apply(initVals);
    expectTensorsClose(postConstraint, tensor1d(new Float32Array([
                         -0.2208630521 / 2, 0.4417261043 / 2, 0,
                         0.8834522086 / 2, -1.104315261 / 2, 1.325178313 / 2
                       ])));
  });
  it('minMaxNorm', () => {
    const constraint = getConstraint('minMaxNorm');
    const postConstraint = constraint.apply(initVals);
    expectTensorsClose(postConstraint, tensor1d(new Float32Array([
                         -0.2208630521 / 2, 0.4417261043 / 2, 0,
                         0.8834522086 / 2, -1.104315261 / 2, 1.325178313 / 2
                       ])));
  });
});

describeMathCPU('constraints.get', () => {
  it('by string', () => {
    const constraint = getConstraint('maxNorm');
    const config = serializeConstraint(constraint) as ConfigDict;
    const nestedConfig = config.config as ConfigDict;
    expect(nestedConfig.maxValue).toEqual(2);
    expect(nestedConfig.axis).toEqual(0);
  });

  it('by string, upper case', () => {
    const constraint = getConstraint('maxNorm');
    const config = serializeConstraint(constraint) as ConfigDict;
    const nestedConfig = config.config as ConfigDict;
    expect(nestedConfig.maxValue).toEqual(2);
    expect(nestedConfig.axis).toEqual(0);
  });

  it('by existing object', () => {
    const origConstraint = tfl.constraints.nonNeg();
    expect(getConstraint(origConstraint)).toEqual(origConstraint);
  });
  it('by config dict', () => {
    const origConstraint = tfl.constraints.minMaxNorm(
        {minValue: 0, maxValue: 2, rate: 3, axis: 4});
    const constraint =
        getConstraint(serializeConstraint(origConstraint) as ConfigDict);
    expect(serializeConstraint(constraint))
        .toEqual(serializeConstraint(origConstraint));
  });
});

describe('Constraints Serialization', () => {
  it('Built-ins', () => {
    // Test both types of captialization.
    const constraints: ConstraintIdentifier[] = [
      'maxNorm', 'nonNeg', 'unitNorm', 'minMaxNorm', 'MaxNorm', 'NonNeg',
      'UnitNorm', 'MinMaxNorm'
    ];
    for (const name of constraints) {
      const constraint = getConstraint(name);
      const config = serializeConstraint(constraint) as ConfigDict;
      const reconstituted = deserializeConstraint(config);
      expect(reconstituted).toEqual(constraint);
    }
  });
});
