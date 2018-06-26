/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {scalar, zeros} from '@tensorflow/tfjs-core';

import {LayerVariable} from '../variables';

import * as variable_utils from './variable_utils';

describe('countParamsInWeights', () => {
  it('Zero weights', () => {
    expect(variable_utils.countParamsInWeights([])).toEqual(0);
  });

  it('One float32 weight', () => {
    const weight1 = new LayerVariable(zeros([2, 3]));
    expect(variable_utils.countParamsInWeights([weight1])).toEqual(6);
  });

  it('One float32 scalar weight', () => {
    const weight1 = new LayerVariable(scalar(42));
    expect(variable_utils.countParamsInWeights([weight1])).toEqual(1);
  });

  it('One int32 weight', () => {
    const weight1 = new LayerVariable(zeros([1, 3, 4], 'int32'), 'int32');
    expect(variable_utils.countParamsInWeights([weight1])).toEqual(12);
  });

  it('Two weights, mixed types and shapes', () => {
    const weight1 = new LayerVariable(scalar(42));
    const weight2 = new LayerVariable(zeros([2, 3]));
    const weight3 = new LayerVariable(zeros([1, 3, 4], 'int32'), 'int32');
    expect(variable_utils.countParamsInWeights([
      weight1, weight2, weight3
    ])).toEqual(19);
  });
});
