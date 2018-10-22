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

import {eye, randomNormal, serialization, Tensor, Tensor2D, tensor2d} from '@tensorflow/tfjs-core';

import * as tfl from './index';
import {checkDistribution, checkFanMode, getInitializer, serializeInitializer, VALID_DISTRIBUTION_VALUES, VALID_FAN_MODE_VALUES, VarianceScaling} from './initializers';
import {deserialize} from './layers/serialization';
import {JsonDict} from './types';
import * as math_utils from './utils/math_utils';
import {convertPythonicToTs} from './utils/serialization_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectNoLeakedTensors, expectTensorsClose, expectTensorsValuesInRange} from './utils/test_utils';


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

  it('Deserialize model containing GlorotUniform initializer', () => {
    // From https://github.com/tensorflow/tfjs/issues/798
    const testModelJSON =
        // tslint:disable-next-line:max-line-length
        `{"modelTopology": {"keras_version": "2.1.6-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "emoji_autoencoder", "layers": [{"name": "input_1", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 128, 128, 1], "dtype": "float32", "sparse": false, "name": "input_1"}, "inbound_nodes": []}, {"name": "Encoder", "class_name": "Model", "config": {"name": "Encoder", "layers": [{"name": "input_128x128", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 128, 128, 1], "dtype": "float32", "sparse": false, "name": "input_128x128"}, "inbound_nodes": []}, {"name": "Convolution1", "class_name": "Conv2D", "config": {"name": "Convolution1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_128x128", 0, 0, {}]]]}, {"name": "shrink_64x64", "class_name": "MaxPooling2D", "config": {"name": "shrink_64x64", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["Convolution1", 0, 0, {}]]]}, {"name": "Convolution2", "class_name": "Conv2D", "config": {"name": "Convolution2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["shrink_64x64", 0, 0, {}]]]}, {"name": "shrink_32x32", "class_name": "MaxPooling2D", "config": {"name": "shrink_32x32", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["Convolution2", 0, 0, {}]]]}, {"name": "Convolution3", "class_name": "Conv2D", "config": {"name": "Convolution3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["shrink_32x32", 0, 0, {}]]]}, {"name": "shrink_8x8", "class_name": "MaxPooling2D", "config": {"name": "shrink_8x8", "trainable": true, "dtype": "float32", "pool_size": [4, 4], "padding": "same", "strides": [4, 4], "data_format": "channels_last"}, "inbound_nodes": [[["Convolution3", 0, 0, {}]]]}, {"name": "conv2d", "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["shrink_8x8", 0, 0, {}]]]}, {"name": "shrink_4x4", "class_name": "MaxPooling2D", "config": {"name": "shrink_4x4", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"name": "matrix-to-vector", "class_name": "Flatten", "config": {"name": "matrix-to-vector", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["shrink_4x4", 0, 0, {}]]]}, {"name": "link_flat_to_64x1", "class_name": "Dense", "config": {"name": "link_flat_to_64x1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["matrix-to-vector", 0, 0, {}]]]}, {"name": "output_8x1", "class_name": "Dense", "config": {"name": "output_8x1", "trainable": true, "dtype": "float32", "units": 8, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["link_flat_to_64x1", 0, 0, {}]]]}], "input_layers": [["input_128x128", 0, 0]], "output_layers": [["output_8x1", 0, 0]]}, "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"name": "Decoder", "class_name": "Model", "config": {"name": "Decoder", "layers": [{"name": "input_8x1", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 8], "dtype": "float32", "sparse": false, "name": "input_8x1"}, "inbound_nodes": []}, {"name": "activate_input", "class_name": "Dense", "config": {"name": "activate_input", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_8x1", 0, 0, {}]]]}, {"name": "link_reshape_64x1", "class_name": "Dense", "config": {"name": "link_reshape_64x1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["activate_input", 0, 0, {}]]]}, {"name": "reshape_8x8", "class_name": "Reshape", "config": {"name": "reshape_8x8", "trainable": true, "dtype": "float32", "target_shape": [8, 8, 1]}, "inbound_nodes": [[["link_reshape_64x1", 0, 0, {}]]]}, {"name": "conv2d_1", "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["reshape_8x8", 0, 0, {}]]]}, {"name": "grow_16x16", "class_name": "UpSampling2D", "config": {"name": "grow_16x16", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"name": "conv2d_2", "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["grow_16x16", 0, 0, {}]]]}, {"name": "grow_32x32", "class_name": "UpSampling2D", "config": {"name": "grow_32x32", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"name": "conv2d_3", "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["grow_32x32", 0, 0, {}]]]}, {"name": "grow_64x64", "class_name": "UpSampling2D", "config": {"name": "grow_64x64", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"name": "conv2d_4", "class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["grow_64x64", 0, 0, {}]]]}, {"name": "grow_128x128", "class_name": "UpSampling2D", "config": {"name": "grow_128x128", "trainable": true, "dtype": "float32", "size": [2, 2], "data_format": "channels_last"}, "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"name": "output_128x128", "class_name": "Conv2D", "config": {"name": "output_128x128", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [5, 5], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["grow_128x128", 0, 0, {}]]]}], "input_layers": [["input_8x1", 0, 0]], "output_layers": [["output_128x128", 0, 0]]}, "inbound_nodes": [[["Encoder", 1, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Decoder", 1, 0]]}}, "training_config": {"optimizer_config": {"class_name": "Adadelta", "config": {"lr": 1.0, "rho": 0.95, "decay": 0.0, "epsilon": 1e-07}}, "loss": "mean_squared_error", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null}}, "weightsManifest": [{"paths": ["group1-shard1of1"], "weights": [{"name": "activate_input/kernel", "shape": [8, 64], "dtype": "float32"}, {"name": "activate_input/bias", "shape": [64], "dtype": "float32"}, {"name": "link_reshape_64x1/kernel", "shape": [64, 64], "dtype": "float32"}, {"name": "link_reshape_64x1/bias", "shape": [64], "dtype": "float32"}, {"name": "conv2d_1/kernel", "shape": [3, 3, 1, 8], "dtype": "float32"}, {"name": "conv2d_1/bias", "shape": [8], "dtype": "float32"}, {"name": "conv2d_2/kernel", "shape": [3, 3, 8, 8], "dtype": "float32"}, {"name": "conv2d_2/bias", "shape": [8], "dtype": "float32"}, {"name": "conv2d_3/kernel", "shape": [3, 3, 8, 8], "dtype": "float32"}, {"name": "conv2d_3/bias", "shape": [8], "dtype": "float32"}, {"name": "conv2d_4/kernel", "shape": [3, 3, 8, 16], "dtype": "float32"}, {"name": "conv2d_4/bias", "shape": [16], "dtype": "float32"}, {"name": "output_128x128/kernel", "shape": [5, 5, 16, 1], "dtype": "float32"}, {"name": "output_128x128/bias", "shape": [1], "dtype": "float32"}, {"name": "Convolution1/kernel", "shape": [5, 5, 1, 16], "dtype": "float32"}, {"name": "Convolution1/bias", "shape": [16], "dtype": "float32"}, {"name": "Convolution2/kernel", "shape": [3, 3, 16, 8], "dtype": "float32"}, {"name": "Convolution2/bias", "shape": [8], "dtype": "float32"}, {"name": "Convolution3/kernel", "shape": [3, 3, 8, 8], "dtype": "float32"}, {"name": "Convolution3/bias", "shape": [8], "dtype": "float32"}, {"name": "conv2d/kernel", "shape": [3, 3, 8, 4], "dtype": "float32"}, {"name": "conv2d/bias", "shape": [4], "dtype": "float32"}, {"name": "link_flat_to_64x1/kernel", "shape": [64, 64], "dtype": "float32"}, {"name": "link_flat_to_64x1/bias", "shape": [64], "dtype": "float32"}, {"name": "output_8x1/kernel", "shape": [64, 8], "dtype": "float32"}, {"name": "output_8x1/bias", "shape": [8], "dtype": "float32"}]}]}`;
    const modelConfig = convertPythonicToTs(
        JSON.parse(testModelJSON).modelTopology.model_config);

    const model = deserialize(modelConfig as JsonDict) as tfl.Model;
    expect(model.layers.length).toEqual(3);
    expect(model.layers[0] instanceof tfl.Model).toEqual(false);
    expect(model.layers[1] instanceof tfl.Model).toEqual(true);
    expect(model.layers[2] instanceof tfl.Model).toEqual(true);
    expect(model.inputs[0].shape).toEqual([null, 128, 128, 1]);
    expect(model.outputs[0].shape).toEqual([null, 128, 128, 1]);
    expect((model.predict(randomNormal([1, 128, 128, 1])) as Tensor).shape)
        .toEqual([1, 128, 128, 1]);
  });
});
