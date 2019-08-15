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
 * Unit tests for optimizers.ts.
 */

import {AdagradOptimizer, AdadeltaOptimizer, AdamOptimizer, AdamaxOptimizer, RMSPropOptimizer, SGDOptimizer} from '@tensorflow/tfjs-core';

import {getOptimizer} from './optimizers';
import {describeMathCPU} from './utils/test_utils';


describeMathCPU('getOptimizer', () => {
  // TODO(nsthorat): Assert defaults by getting config from the optimizer.

  it(`can instantiate SGD`, () => {
    const optimizer = getOptimizer('SGD');
    expect(optimizer instanceof SGDOptimizer).toBe(true);
  });
  it(`can instantiate sgd`, () => {
    const optimizer = getOptimizer('sgd');
    expect(optimizer instanceof SGDOptimizer).toBe(true);
  });
  it(`can instantiate Adam`, () => {
    const optimizer = getOptimizer('Adam');
    expect(optimizer instanceof AdamOptimizer).toBe(true);
  });
  it(`can instantiate adam`, () => {
    const optimizer = getOptimizer('adam');
    expect(optimizer instanceof AdamOptimizer).toBe(true);
  });
  it(`can instantiate RMSProp`, () => {
    const optimizer = getOptimizer('RMSProp');
    expect(optimizer instanceof RMSPropOptimizer).toBe(true);
  });
  it(`can instantiate rmsprop`, () => {
    const optimizer = getOptimizer('rmsprop');
    expect(optimizer instanceof RMSPropOptimizer).toBe(true);
  });
  it(`can instantiate Adagrad`, () => {
    const optimizer = getOptimizer('Adagrad');
    expect(optimizer instanceof AdagradOptimizer).toBe(true);
  });
  it(`can instantiate adagrad`, () => {
    const optimizer = getOptimizer('adagrad');
    expect(optimizer instanceof AdagradOptimizer).toBe(true);
  });
  it(`can instantiate Adadelta`, () => {
    const optimizer = getOptimizer('Adadelta');
    expect(optimizer instanceof AdadeltaOptimizer).toBe(true);
  });
  it(`can instantiate adadelta`, () => {
    const optimizer = getOptimizer('adadelta');
    expect(optimizer instanceof AdadeltaOptimizer).toBe(true);
  });
  it(`can instantiate Adamax`, () => {
    const optimizer = getOptimizer('Adamax');
    expect(optimizer instanceof AdamaxOptimizer).toBe(true);
  });
  it(`can instantiate adamax`, () => {
    const optimizer = getOptimizer('adamax');
    expect(optimizer instanceof AdamaxOptimizer).toBe(true);
  });
  it('throws for non-existent optimizer', () => {
    expect(() => getOptimizer('not an optimizer'))
      .toThrowError(/Unknown Optimizer/);
  });

});
