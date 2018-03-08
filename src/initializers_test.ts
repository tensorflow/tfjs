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
import {tensor2d} from 'deeplearn';

import {Distribution, FanMode, get, GlorotUniform, serialize, Zeros} from './initializers';
import {DType} from './types';
import {ConfigDict} from './types';
import * as math_utils from './utils/math_utils';
import {describeMathCPU, expectTensorsClose, expectTensorsValuesInRange} from './utils/test_utils';

// tslint:enable:max-line-length

describeMathCPU('Zeros initializer', () => {
  it('1D', () => {
    const init = get('Zeros');
    const weights = init.apply([3], DType.float32);
    expect(weights.shape).toEqual([3]);
    expect(weights.dtype).toEqual(DType.float32);
    expect(weights.dataSync()).toEqual(new Float32Array([0, 0, 0]));
  });

  it('2D', () => {
    const init = get('Zeros');
    const weights = init.apply([2, 2], DType.float32);
    expect(weights.shape).toEqual([2, 2]);
    expect(weights.dtype).toEqual(DType.float32);
    expect(weights.dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
  });
});

describeMathCPU('Ones initializer', () => {
  it('1D', () => {
    const init = get('Ones');
    const weights = init.apply([3], DType.float32);
    expect(weights.shape).toEqual([3]);
    expect(weights.dtype).toEqual(DType.float32);
    expect(weights.dataSync()).toEqual(new Float32Array([1, 1, 1]));
  });

  it('2D', () => {
    const init = get('Ones');
    const weights = init.apply([2, 2], DType.float32);
    expect(weights.shape).toEqual([2, 2]);
    expect(weights.dtype).toEqual(DType.float32);
    expect(weights.dataSync()).toEqual(new Float32Array([1, 1, 1, 1]));
  });
});

describeMathCPU('Constant initializer', () => {
  it('1D', () => {
    const initializerConfig:
        ConfigDict = {className: 'Constant', config: {value: 5}};
    const init = get(initializerConfig);
    const weights = init.apply([3], DType.float32);
    expect(weights.shape).toEqual([3]);
    expect(weights.dtype).toEqual(DType.float32);
    expect(weights.dataSync()).toEqual(new Float32Array([5, 5, 5]));
  });

  it('2D', () => {
    const initializerConfig:
        ConfigDict = {className: 'Constant', config: {value: 5}};
    const init = get(initializerConfig);
    const weights = init.apply([2, 2], DType.float32);
    expect(weights.shape).toEqual([2, 2]);
    expect(weights.dtype).toEqual(DType.float32);
    expect(weights.dataSync()).toEqual(new Float32Array([5, 5, 5, 5]));
  });
});

describeMathCPU('Identity initializer', () => {
  it('1D', () => {
    const initializerConfig:
        ConfigDict = {className: 'Identity', config: {gain: 5}};
    const init = get(initializerConfig);
    expect(() => {
      init.apply([4]);
    }).toThrowError(/2D square/);
  });

  it('2D', () => {
    const initializerConfig:
        ConfigDict = {className: 'Identity', config: {gain: 5}};
    const init = get(initializerConfig);
    const weights = init.apply([2, 2], DType.float32);
    expect(weights.shape).toEqual([2, 2]);
    expect(weights.dtype).toEqual(DType.float32);
    expectTensorsClose(weights, tensor2d([5, 0, 0, 5], [2, 2]));
  });
});


describeMathCPU('RandomUniform initializer', () => {
  const shape = [7, 2];
  it('default', () => {
    const init = get('RandomUniform');
    const weights = init.apply(shape, DType.float32);
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual(DType.float32);
    expectTensorsValuesInRange(weights, -0.05, 0.05);
  });

  it('with configured min max val', () => {
    const initializerConfig: ConfigDict = {
      className: 'RandomUniform',
      config: {minval: 17, maxval: 47}
    };
    const init = get(initializerConfig);
    const weights = init.apply(shape, DType.float32);
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual(DType.float32);
    expectTensorsValuesInRange(weights, 17, 47);
  });
});

describeMathCPU('RandomNormal initializer', () => {
  const shape = [7, 2];
  it('default', () => {
    const init = get('RandomNormal');
    const weights = init.apply(shape, DType.float32);
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual(DType.float32);
    // TODO(bileschi): Add test to assert the values match expectations.
  });

  it('with configured min max val', () => {
    const initializerConfig: ConfigDict = {
      className: 'RandomNormal',
      config: {mean: 1.0, stddev: 0.001}
    };
    const init = get(initializerConfig);
    const weights = init.apply(shape, DType.float32);
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual(DType.float32);
    // TODO(bileschi): Add test to assert the values match expectations.
  });
});

describeMathCPU('HeNormal initializer', () => {
  const shape = [7, 2];
  const stddev = Math.sqrt(2 / shape[0]);
  it('default', () => {
    const init = get('HeNormal');
    const weights = init.apply(shape, DType.float32);
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual(DType.float32);
    expectTensorsValuesInRange(weights, -2 * stddev, 2 * stddev);
  });
});

describeMathCPU('LecunNormal initializer', () => {
  const shape = [7, 2];
  const stddev = Math.sqrt(1 / shape[0]);
  it('default', () => {
    const init = get('LeCunNormal');
    const weights = init.apply(shape, DType.float32);
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual(DType.float32);
    expectTensorsValuesInRange(weights, -2 * stddev, 2 * stddev);
  });
});

describeMathCPU('TruncatedNormal initializer', () => {
  const shape = [7, 2];
  it('default', () => {
    const init = get('TruncatedNormal');
    const weights = init.apply(shape, DType.float32);
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual(DType.float32);
    expectTensorsValuesInRange(weights, -0.1, 0.1);
  });

  it('with configured min max val', () => {
    const initializerConfig: ConfigDict = {
      className: 'TruncatedNormal',
      config: {mean: 1.0, stddev: 0.5}
    };
    const init = get(initializerConfig);
    const weights = init.apply(shape, DType.float32);
    expect(weights.shape).toEqual(shape);
    expect(weights.dtype).toEqual(DType.float32);
    expectTensorsValuesInRange(weights, 0.0, 2.0);
  });
});

describeMathCPU('Glorot uniform initializer', () => {
  it('1D', () => {
    const init = get('GlorotUniform');
    let weights = init.apply([3], DType.float32);
    expect(weights.shape).toEqual([3]);
    expect(weights.dtype).toEqual(DType.float32);
    let scale = 1 / ((Math.sqrt(3) + Math.sqrt(3)) / 2);
    let limit = Math.sqrt(3 * scale);
    expect(math_utils.max(weights.dataSync() as Float32Array))
        .toBeLessThan(limit);
    expect(math_utils.min(weights.dataSync() as Float32Array))
        .toBeGreaterThan(-limit);

    weights = init.apply([30], DType.float32);
    expect(weights.shape).toEqual([30]);
    expect(weights.dtype).toEqual(DType.float32);
    scale = 1 / ((Math.sqrt(30) + Math.sqrt(30)) / 2);
    limit = Math.sqrt(3 * scale);
    expect(math_utils.max(weights.dataSync() as Float32Array))
        .toBeLessThan(limit);
    expect(math_utils.min(weights.dataSync() as Float32Array))
        .toBeGreaterThan(-limit);
  });

  it('2D', () => {
    const init = get('GlorotUniform');
    let weights = init.apply([2, 2], DType.float32);
    expect(weights.shape).toEqual([2, 2]);
    expect(weights.dtype).toEqual(DType.float32);
    let scale = 1 / ((Math.sqrt(2) + Math.sqrt(2)) / 2);
    let limit = Math.sqrt(3 * scale);
    expect(math_utils.max(weights.dataSync() as Float32Array))
        .toBeLessThan(limit);
    expect(math_utils.min(weights.dataSync() as Float32Array))
        .toBeGreaterThan(-limit);

    weights = init.apply([20, 20], DType.float32);
    expect(weights.shape).toEqual([20, 20]);
    expect(weights.dtype).toEqual(DType.float32);
    scale = 1 / ((Math.sqrt(20) + Math.sqrt(20)) / 2);
    limit = Math.sqrt(3 * scale);
    expect(math_utils.max(weights.dataSync() as Float32Array))
        .toBeLessThan(limit);
    expect(math_utils.min(weights.dataSync() as Float32Array))
        .toBeGreaterThan(-limit);
  });
});

describeMathCPU('Glorot normal initializer', () => {
  it('1D', () => {
    const init = get('GlorotNormal');
    let weights = init.apply([30], DType.float32);
    expect(weights.shape).toEqual([30]);
    expect(weights.dtype).toEqual(DType.float32);
    const variance1 = math_utils.variance(weights.dataSync() as Float32Array);

    weights = init.apply([120], DType.float32);
    expect(weights.shape).toEqual([120]);
    expect(weights.dtype).toEqual(DType.float32);
    const variance2 = math_utils.variance(weights.dataSync() as Float32Array);

    expect(variance2).toBeLessThan(variance1);
  });

  it('2D', () => {
    const init = get('GlorotNormal');
    let weights = init.apply([5, 6], DType.float32);
    expect(weights.shape).toEqual([5, 6]);
    expect(weights.dtype).toEqual(DType.float32);
    const variance1 = math_utils.variance(weights.dataSync() as Float32Array);

    weights = init.apply([10, 12], DType.float32);
    expect(weights.shape).toEqual([10, 12]);
    expect(weights.dtype).toEqual(DType.float32);
    const variance2 = math_utils.variance(weights.dataSync() as Float32Array);

    expect(variance2).toBeLessThan(variance1);
  });
});

describeMathCPU('initializers.get', () => {
  it('by string', () => {
    const initializer = get('GlorotNormal');
    const config = serialize(initializer) as ConfigDict;
    const nestedConfig = config.config as ConfigDict;
    expect(nestedConfig.scale).toEqual(1.0);
    expect(nestedConfig.mode).toEqual(FanMode.FAN_AVG);
    expect(nestedConfig.distribution).toEqual(Distribution.NORMAL);
  });
  it('by existing object', () => {
    const origInit = new Zeros();
    const initializer = get(origInit);
    expect(initializer).toEqual(origInit);
  });
  it('by config dict', () => {
    const origInit = new GlorotUniform({seed: 10});
    const initializer = get(serialize(origInit) as ConfigDict);
    expect(serialize(initializer)).toEqual(serialize(origInit));
  });
});

describe('Invalid intializer identifier', () => {
  it('Throws exception', () => {
    expect(() => {
      get('invalid_initializer_id');
    }).toThrowError();
  });
});
