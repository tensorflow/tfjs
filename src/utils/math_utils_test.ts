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
 * Unit tests for math_utils.
 */

import * as math_utils from './math_utils';
import {describeMathCPU} from './test_utils';

describe('isInteger', () => {
  it('True cases', () => {
    expect(math_utils.isInteger(-103)).toBe(true);
    expect(math_utils.isInteger(0)).toBe(true);
    expect(math_utils.isInteger(1337)).toBe(true);
  });

  it('False cases', () => {
    expect(math_utils.isInteger(-1.03)).toBe(false);
    expect(math_utils.isInteger(0.008)).toBe(false);
    expect(math_utils.isInteger(133.7)).toBe(false);
  });
});

describe('arrayProd', () => {
  it('Full length', () => {
    expect(math_utils.arrayProd([2, 3, 4])).toEqual(24);
    expect(math_utils.arrayProd(new Float32Array([2, 3, 4]))).toEqual(24);
  });

  it('Partial from beginning', () => {
    expect(math_utils.arrayProd([2, 3, 4], null, 2)).toEqual(6);
    expect(math_utils.arrayProd([2, 3, 4], 0, 2)).toEqual(6);
  });

  it('Partial to end', () => {
    expect(math_utils.arrayProd([2, 3, 4], 1)).toEqual(12);
    expect(math_utils.arrayProd([2, 3, 4], 1, 3)).toEqual(12);
  });

  it('Partial no beginninng no end', () => {
    expect(math_utils.arrayProd([2, 3, 4, 5], 1, 3)).toEqual(12);
  });

  it('Empty array', () => {
    expect(math_utils.arrayProd([])).toEqual(1);
  });
});

describeMathCPU('min', () => {
  it('Number array', () => {
    expect(math_utils.min([-100, -200, 150])).toEqual(-200);
  });

  it('Float32Array', () => {
    expect(math_utils.min(new Float32Array([-100, -200, 150]))).toEqual(-200);
  });
});

describeMathCPU('max', () => {
  it('Number array', () => {
    expect(math_utils.max([-100, -200, 150])).toEqual(150);
  });

  it('Float32Array', () => {
    expect(math_utils.max(new Float32Array([-100, -200, 150]))).toEqual(150);
  });
});

describeMathCPU('sum', () => {
  it('Number array', () => {
    expect(math_utils.sum([-100, -200, 150])).toEqual(-150);
  });

  it('Float32Array', () => {
    expect(math_utils.sum(new Float32Array([-100, -200, 150]))).toEqual(-150);
  });
});

describeMathCPU('mean', () => {
  it('Number array', () => {
    expect(math_utils.mean([-100, -200, 150])).toEqual(-50);
  });

  it('Float32Array', () => {
    expect(math_utils.mean(new Float32Array([-100, -200, 150]))).toEqual(-50);
  });
});


describeMathCPU('variance', () => {
  it('Number array', () => {
    expect(math_utils.variance([-100, -200, 150, 50])).toEqual(18125);
  });

  it('Float32Array', () => {
    expect(math_utils.variance(new Float32Array([-100, -200, 150, 50])))
        .toEqual(18125);
  });
});

describe('range', () => {
  it('end > begin', () => {
    expect(math_utils.range(0, 1)).toEqual([0]);
    expect(math_utils.range(0, 5)).toEqual([0, 1, 2, 3, 4]);
    expect(math_utils.range(-10, -5)).toEqual([-10, -9, -8, -7, -6]);
    expect(math_utils.range(-3, 3)).toEqual([-3, -2, -1, 0, 1, 2]);
  });
  it('end === begin', () => {
    expect(math_utils.range(0, 0)).toEqual([]);
    expect(math_utils.range(-2, -2)).toEqual([]);
  });
  it('end < begin throws error', () => {
    expect(() => math_utils.range(0, -2)).toThrowError(/.*-2.*0.*forbidden/);
  });
});
