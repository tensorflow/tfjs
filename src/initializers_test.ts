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
 * Unit tests for initializers.
 */

// tslint:disable:max-line-length
import {eye, serialization, Tensor2D, tensor2d} from '@tensorflow/tfjs-core';

import * as tfl from './index';
import {checkDistribution, checkFanMode, getInitializer, serializeInitializer, VALID_DISTRIBUTION_VALUES, VALID_FAN_MODE_VALUES, VarianceScaling} from './initializers';
import * as math_utils from './utils/math_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectNoLeakedTensors, expectTensorsClose, expectTensorsValuesInRange} from './utils/test_utils';

// tslint:enable:max-line-length

describeMathCPU('Zeros initializer', () => {
  it('1D', () => {
    const init = getInitializer('zeros');
    const weights = init.apply([3], 'float32');
    expect(weights.shape).toEqual([3]);
    expect(weights.dtype).toEqual('float32');
    expect(weights.dataSync()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('1D, upper case', () => {
    const init = getInitializer('Zeros');
    const weights = init.apply([3], 'float32');
    expect(weights.shape).toEqual([3]);
    expect(weights.dtype).toEqual('float32');
    expect(weights.dataSync()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('2D', () => {
    const init = getInitializer('zeros');
    const weights = init.apply([2, 2], 'float32');
    expect(weights.shape).toEqual([2, 2]);
    expect(weights.dtype).toEqual('float32');
    expect(weights.dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('Does not leak', () => {
    expectNoLeakedTensors(() => getInitializer('zeros').apply([3]), 1);
  });
});

describeMathCPU('Ones initializer', () => {
  it('1D', () => {
    const init = getInitializer('ones');
    const weights = init.apply([3], 'float32');
    expect(weights.shape).toEqual([3]);
    expect(weights.dtype).toEqual('float32');
    expect(weights.dataSync()).toEqual(new Float32Array([1, 1, 1]));
  });

  it('1D, upper case', () => {
    const init = getInitializer('Ones');
    const weights = init.apply([3], 'float32');
    expect(weights.shape).toEqual([3]);
    expect(weights.dtype).toEqual('float32');
    expect(weights.dataSync()).toEqual(new Float32Array([1, 1, 1]));
  });

  it('2D', () => {
    const init = getInitializer('ones');
    const weights = init.apply([2, 2], 'float32');
    expect(weights.shape).toEqual([2, 2]);
    expect(weights.dtype).toEqual('float32');
    expect(weights.dataSync()).toEqual(new Float32Array([1, 1, 1, 1]));
  });
  it('Does not leak', () => {
    expectNoLeakedTensors(() => getInitializer('ones').apply([3]), 1);
  });
});

describeMathCPU('Constant initializer', () => {
  it('1D, from config dict', () => {
    const initializerConfig:
        serialization.ConfigDict = {className: 'Constant', config: {value: 5}};
    const init = getInitializer(initializerConfig);
    const weights = init.apply([3], 'float32');
    expect(weights.shape).toEqual([3]);
    expect(weights.dtype).toEqual('float32');
    expect(weights.dataSync()).toEqual(new Float32Array([5, 5, 5]));
  });

  it('1D, from builder function', () => {
    const init = tfl.initializers.constant({value: 5});
    const weights = init.apply([3], 'float32');
    expect(weights.shape).toEqual([3]);
    expect(weights.dtype).toEqual('float32');
    expect(weights.dataSync()).toEqual(new Float32Array([5, 5, 5]));
  });

  it('1D, from builder function: passing a direct value throws error', () => {
    // tslint:disable-next-line:no-any
    expect(() => tfl.initializers.constant(5 as any))
        .toThrowError(/Expected.*ConstantConfig/);
  });

  it('2D, from config dict', () => {
    const initializerConfig:
        serialization.ConfigDict = {className: 'Constant', config: {value: 5}};
    const init = getInitializer(initializerConfig);
    const weights = init.apply([2, 2], 'float32');
    expect(weights.shape).toEqual([2, 2]);
    expect(weights.dtype).toEqual('float32');
    expect(weights.dataSync()).toEqual(new Float32Array([5, 5, 5, 5]));
  });

  it('Does not leak', () => {
    const initializerConfig:
        serialization.ConfigDict = {className: 'Constant', config: {value: 5}};
    expectNoLeakedTensors(
        () => getInitializer(initializerConfig).apply([3]), 1);
  });
});

describeMathCPU('Identity initializer', () => {
  it('1D', () => {
    const initializerConfig:
        serialization.ConfigDict = {className: 'Identity', config: {gain: 5}};
    const init = getInitializer(initializerConfig);
    expect(() => init.apply([4])).toThrowError(/2D square/);
  });

  it('1D, from config', () => {
    const initializerConfig:
        serialization.ConfigDict = {className: 'Identity', config: {gain: 5}};
    const init = getInitializer(initializerConfig);
    expect(() => init.apply([4])).toThrowError(/2D square/);
  });

  it('2D', () => {
    const initializerConfig:
        serialization.ConfigDict = {className: 'Identity', config: {gain: 5}};
    const init = getInitializer(initializerConfig);
    const weights = init.apply([2, 2], 'float32');
    expect(weights.shape).toEqual([2, 2]);
    expect(weights.dtype).toEqual('float32');
    expectTensorsClose(weights, tensor2d([5, 0, 0, 5], [2, 2]));
  });
});


describeMathCPU('RandomUniform initializer', () => {
  const shape = [7, 2];
  it('default', () => {
    const init = getInitializer('randomUniform');
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    expectTensorsValuesInRange(weights, -0.05, 0.05);
  });

  it('default, upper case', () => {
    const init = getInitializer('RandomUniform');
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    expectTensorsValuesInRange(weights, -0.05, 0.05);
  });

  it('with configured min max val', () => {
    const initializerConfig: serialization.ConfigDict = {
      className: 'RandomUniform',
      config: {minval: 17, maxval: 47}
    };
    const init = getInitializer(initializerConfig);
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    expectTensorsValuesInRange(weights, 17, 47);
  });
  it('Does not leak', () => {
    expectNoLeakedTensors(() => getInitializer('RandomUniform').apply([3]), 1);
  });
});

describeMathCPU('RandomNormal initializer', () => {
  const shape = [7, 2];
  it('default', () => {
    const init = getInitializer('randomNormal');
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    // TODO(bileschi): Add test to assert the values match expectations.
  });

  it('default, upper case', () => {
    const init = getInitializer('RandomNormal');
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    // TODO(bileschi): Add test to assert the values match expectations.
  });

  it('with configured min max val', () => {
    const initializerConfig: serialization.ConfigDict = {
      className: 'RandomNormal',
      config: {mean: 1.0, stddev: 0.001}
    };
    const init = getInitializer(initializerConfig);
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    // TODO(bileschi): Add test to assert the values match expectations.
  });
  it('Does not leak', () => {
    expectNoLeakedTensors(() => getInitializer('RandomNormal').apply([3]), 1);
  });
});

describeMathCPU('HeNormal initializer', () => {
  const shape = [7, 2];
  const stddev = Math.sqrt(2 / shape[0]);
  it('default', () => {
    const init = getInitializer('heNormal');
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    expectTensorsValuesInRange(weights, -2 * stddev, 2 * stddev);
    expect(init.getClassName()).toEqual(VarianceScaling.className);
  });

  it('default, upper case', () => {
    const init = getInitializer('HeNormal');
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    expectTensorsValuesInRange(weights, -2 * stddev, 2 * stddev);
  });
  it('Does not leak', () => {
    expectNoLeakedTensors(() => getInitializer('HeNormal').apply([3]), 1);
  });
});

describeMathCPU('LecunNormal initializer', () => {
  const shape = [7, 2];
  const stddev = Math.sqrt(1 / shape[0]);
  it('default', () => {
    const init = getInitializer('leCunNormal');
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    expectTensorsValuesInRange(weights, -2 * stddev, 2 * stddev);
    expect(init.getClassName()).toEqual(VarianceScaling.className);
  });

  it('default, upper case', () => {
    const init = getInitializer('LeCunNormal');
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    expectTensorsValuesInRange(weights, -2 * stddev, 2 * stddev);
  });
  it('Does not leak', () => {
    expectNoLeakedTensors(() => getInitializer('LeCunNormal').apply([3]), 1);
  });
});

describeMathCPU('TruncatedNormal initializer', () => {
  const shape = [7, 2];
  it('default', () => {
    const init = getInitializer('truncatedNormal');
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    expectTensorsValuesInRange(weights, -0.1, 0.1);
  });

  it('default, upper case', () => {
    const init = getInitializer('TruncatedNormal');
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    expectTensorsValuesInRange(weights, -0.1, 0.1);
  });

  it('with configured min max val', () => {
    const initializerConfig: serialization.ConfigDict = {
      className: 'TruncatedNormal',
      config: {mean: 1.0, stddev: 0.5}
    };
    const init = getInitializer(initializerConfig);
    const weights = init.apply(shape, 'float32');
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual('float32');
    expectTensorsValuesInRange(weights, 0.0, 2.0);
  });
  it('Does not leak', () => {
    expectNoLeakedTensors(
        () => getInitializer('TruncatedNormal').apply([3]), 1);
  });
});

describeMathCPU('Glorot uniform initializer', () => {
  ['glorotUniform', 'GlorotUniform'].forEach(initializer => {
    it('1D ' + initializer, () => {
      const init = getInitializer(initializer);
      let weights = init.apply([3], 'float32');
      expect(weights.shape).toEqual([3]);
      expect(weights.dtype).toEqual('float32');
      let scale = 1 / ((Math.sqrt(3) + Math.sqrt(3)) / 2);
      let limit = Math.sqrt(3 * scale);
      expect(math_utils.max(weights.dataSync() as Float32Array))
          .toBeLessThan(limit);
      expect(math_utils.min(weights.dataSync() as Float32Array))
          .toBeGreaterThan(-limit);

      weights = init.apply([30], 'float32');
      expect(weights.shape).toEqual([30]);
      expect(weights.dtype).toEqual('float32');
      scale = 1 / ((Math.sqrt(30) + Math.sqrt(30)) / 2);
      limit = Math.sqrt(3 * scale);
      expect(math_utils.max(weights.dataSync() as Float32Array))
          .toBeLessThan(limit);
      expect(math_utils.min(weights.dataSync() as Float32Array))
          .toBeGreaterThan(-limit);
      expect(init.getClassName()).toEqual(VarianceScaling.className);
    });

    it('2D ' + initializer, () => {
      const init = getInitializer(initializer);
      let weights = init.apply([2, 2], 'float32');
      expect(weights.shape).toEqual([2, 2]);
      expect(weights.dtype).toEqual('float32');
      let scale = 1 / ((Math.sqrt(2) + Math.sqrt(2)) / 2);
      let limit = Math.sqrt(3 * scale);
      expect(math_utils.max(weights.dataSync() as Float32Array))
          .toBeLessThan(limit);
      expect(math_utils.min(weights.dataSync() as Float32Array))
          .toBeGreaterThan(-limit);

      weights = init.apply([20, 20], 'float32');
      expect(weights.shape).toEqual([20, 20]);
      expect(weights.dtype).toEqual('float32');
      scale = 1 / ((Math.sqrt(20) + Math.sqrt(20)) / 2);
      limit = Math.sqrt(3 * scale);
      expect(math_utils.max(weights.dataSync() as Float32Array))
          .toBeLessThan(limit);
      expect(math_utils.min(weights.dataSync() as Float32Array))
          .toBeGreaterThan(-limit);
    });
  });
  it('Does not leak', () => {
    expectNoLeakedTensors(() => getInitializer('GlorotUniform').apply([3]), 1);
  });
});

describeMathCPU('Glorot normal initializer', () => {
  ['glorotNormal', 'GlorotNormal'].forEach(initializer => {
    it('1D ' + initializer, () => {
      const init = getInitializer(initializer);
      const NUM_TRIALS = 4;
      const varianceArr1: number[] = [];
      const varianceArr2: number[] = [];

      for (let i = 0; i < NUM_TRIALS; ++i) {
        let weights = init.apply([30], 'float32');
        expect(weights.shape).toEqual([30]);
        expect(weights.dtype).toEqual('float32');
        varianceArr1.push(
            math_utils.variance(weights.dataSync() as Float32Array));

        weights = init.apply([1200], 'float32');
        expect(weights.shape).toEqual([1200]);
        expect(weights.dtype).toEqual('float32');
        varianceArr2.push(
            math_utils.variance(weights.dataSync() as Float32Array));
        expect(init.getClassName()).toEqual(VarianceScaling.className);
      }

      const variance1 = math_utils.median(varianceArr1);
      const variance2 = math_utils.median(varianceArr2);
      expect(variance2).toBeLessThan(variance1);
    });

    it('2D ' + initializer, () => {
      const init = getInitializer(initializer);
      const NUM_TRIALS = 4;
      const varianceArr1: number[] = [];
      const varianceArr2: number[] = [];

      for (let i = 0; i < NUM_TRIALS; ++i) {
        let weights = init.apply([5, 6], 'float32');
        expect(weights.shape).toEqual([5, 6]);
        expect(weights.dtype).toEqual('float32');
        varianceArr1.push(
            math_utils.variance(weights.dataSync() as Float32Array));

        weights = init.apply([30, 50], 'float32');
        expect(weights.shape).toEqual([30, 50]);
        expect(weights.dtype).toEqual('float32');
        varianceArr2.push(
            math_utils.variance(weights.dataSync() as Float32Array));
      }

      const variance1 = math_utils.median(varianceArr1);
      const variance2 = math_utils.median(varianceArr2);
      expect(variance2).toBeLessThan(variance1);
    });
  });
  it('Does not leak', () => {
    expectNoLeakedTensors(() => getInitializer('GlorotNormal').apply([3]), 1);
  });
});

describeMathCPU('initializers.get', () => {
  it('by string', () => {
    const initializer = getInitializer('glorotNormal');
    const config =
        serializeInitializer(initializer) as serialization.ConfigDict;
    const nestedConfig = config.config as serialization.ConfigDict;
    expect(nestedConfig.scale).toEqual(1.0);
    expect(nestedConfig.mode).toEqual('fanAvg');
    expect(nestedConfig.distribution).toEqual('normal');
  });
  it('by existing object', () => {
    const origInit = tfl.initializers.zeros();
    const initializer = getInitializer(origInit);
    expect(initializer).toEqual(origInit);
  });
  it('by config dict', () => {
    const origInit = tfl.initializers.glorotUniform({seed: 10});
    const initializer = getInitializer(
        serializeInitializer(origInit) as serialization.ConfigDict);
    expect(serializeInitializer(initializer))
        .toEqual(serializeInitializer(origInit));
  });
});

describe('Invalid intializer identifier', () => {
  it('Throws exception', () => {
    expect(() => {
      getInitializer('invalid_initializer_id');
    }).toThrowError();
  });
});


describe('checkFanMode', () => {
  it('Valid values', () => {
    const extendedValues = VALID_FAN_MODE_VALUES.concat([undefined, null]);
    for (const validValue of extendedValues) {
      // Using implicit "expect().toNotThrow()" for valid values
      checkFanMode(validValue);
    }
  });
  it('Invalid values', () => {
    // Test invalid values are rejected, and reported in the error.
    expect(() => checkFanMode('foo')).toThrowError(/foo/);
    try {
      checkFanMode('bad');
    } catch (e) {
      expect(e).toMatch('FanMode');
      // Test that the error message contains the list of valid values.
      for (const validValue of VALID_FAN_MODE_VALUES) {
        expect(e).toMatch(validValue);
      }
    }
  });
});

describe('checkDistribution', () => {
  it('Valid values', () => {
    const extendedValues = VALID_DISTRIBUTION_VALUES.concat([undefined, null]);
    for (const validValue of extendedValues) {
      // Using implicit "expect().toNotThrow()" for valid values
      checkDistribution(validValue);
    }
  });
  it('Invalid values', () => {
    // Test invalid values are rejected, and reported in the error.
    expect(() => checkDistribution('foo')).toThrowError(/foo/);
    try {
      checkDistribution('bad');
    } catch (e) {
      expect(e).toMatch('Distribution');
      // Test that the error message contains the list of valid values.
      for (const validValue of VALID_DISTRIBUTION_VALUES) {
        expect(e).toMatch(validValue);
      }
    }
  });
});

describeMathCPUAndGPU('Orthogonal Initializer', () => {
  it('2x2', () => {
    const init = getInitializer('Orthogonal');
    const w = init.apply([2, 2], 'float32') as Tensor2D;
    expect(w.shape).toEqual([2, 2]);
    expect(w.dtype).toEqual('float32');
    // Assert that columns of w are orthogonal (w is a unitary matrix).
    expectTensorsClose(w.transpose().matMul(w), eye(2));
  });

  it('1x1 with gain', () => {
    const init = tfl.initializers.orthogonal({gain: 3});
    const w = init.apply([1, 1], 'float32') as Tensor2D;
    expect(w.shape).toEqual([1, 1]);
    expect(w.dtype).toEqual('float32');
    // Assert that columns of w are orthogonal (w is a unitary matrix) and the
    // gain has been reflected.
    expectTensorsClose(w.transpose().matMul(w), tensor2d([[9]], [1, 1]));
  });

  it('4x2', () => {
    const init = getInitializer('Orthogonal');
    const w = init.apply([4, 2], 'float32') as Tensor2D;
    expect(w.shape).toEqual([4, 2]);
    expect(w.dtype).toEqual('float32');
    // Assert that columns of w are orthogonal.
    expectTensorsClose(w.transpose().matMul(w), eye(2));
  });

  it('2x4', () => {
    const init = getInitializer('Orthogonal');
    const w = init.apply([2, 4], 'float32') as Tensor2D;
    expect(w.shape).toEqual([2, 4]);
    expect(w.dtype).toEqual('float32');
    // Assert that columns of w are orthogonal.
    expectTensorsClose(w.matMul(w.transpose()), eye(2));
  });

  it('64x64', () => {
    // Disable console warning during this test.
    // Silence the large-size orthogonal matrix warnings.
    spyOn(console, 'warn').and.callFake((message: string) => {});
    const n = 64;
    const init = getInitializer('Orthogonal');
    const w = init.apply([n, n], 'float32') as Tensor2D;
    expect(w.shape).toEqual([n, n]);
    expect(w.dtype).toEqual('float32');
    // Assert that columns of w are orthogonal.
    expectTensorsClose(w.matMul(w.transpose()), eye(n));
  });
  it('Does not leak', () => {
    const init = getInitializer('Orthogonal');
    expectNoLeakedTensors(() => init.apply([3, 3]), 1);
  });
});
