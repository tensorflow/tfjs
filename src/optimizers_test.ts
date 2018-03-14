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

// tslint:disable:max-line-length
import {Scalar, scalar, Tensor, tensor1d, tensor2d, train} from 'deeplearn';

import * as K from './backend/deeplearnjs_backend';
import {Dense} from './layers/core';
import * as metrics from './metrics';
import {Adagrad, Adam, get as getOptimizer, RMSProp, SGD} from './optimizers';
import {LayerVariable} from './types';
import {describeMathCPU, describeMathCPUAndGPU, describeMathGPU, expectTensorsClose} from './utils/test_utils';
// tslint:enable

/**
 * A mock function to use instead of K.gradients(). Returns loss * ones array *
 * -1^index for each param (variable) passed in, using the same shape as the
 * param passed in.
 * @param loss Loss. This value is actually multiplied by the gradients.
 * @param params The variables to generate mock gradients for.
 */
function mockGradients(
    lossFn: () => Scalar, params: LayerVariable[]): Tensor[] {
  const value = lossFn().get();
  return params.map(
      (p, index) => K.scalarTimesArray(
          scalar(value * Math.pow(-1, index)), K.ones(K.shape(p))));
}

describeMathCPUAndGPU('SGD', () => {
  // Install a mock K.gradients function.
  beforeEach(() => {
    spyOn(K, 'gradients').and.callFake(mockGradients);
  });

  function createVariables(): LayerVariable[] {
    return [K.zerosVariable([]), K.onesVariable([])];
  }

  it('initializes its member variables to default values.', () => {
    const optimizer = new SGD({});
    expect(optimizer.lr).toEqual(0.01);
    expect(optimizer.momentum).toEqual(0.0);
    expect(optimizer.decay).toEqual(0.0);
    expect(optimizer.nesterov).toEqual(false);
  });

  it('initializes its member variables if specified.', () => {
    const lr = 0.2;
    const momentum = 0.0;
    const decay = 0.0;
    const nesterov = false;
    const optimizer = new SGD({lr, momentum, decay, nesterov});
    expect(optimizer.lr).toEqual(lr);
    expect(optimizer.momentum).toEqual(momentum);
    expect(optimizer.decay).toEqual(decay);
    expect(optimizer.nesterov).toEqual(nesterov);
  });

  it('validates lr.', () => {
    const lr = -1;
    expect(() => new SGD({lr})).toThrowError(/Invalid lr/);
  });

  it('validates momentum.', () => {
    const momentum = -1;
    expect(() => new SGD({momentum})).toThrowError(/Invalid momentum/);
  });

  it('validates decay.', () => {
    const decay = -1;
    expect(() => new SGD({decay})).toThrowError(/Invalid decay/);
  });

  const fromConfigValues = [false, true];
  for (const fromConfig of fromConfigValues) {
    const testTitle =
        `updates variables with no decay or momentum: fromConfig=${fromConfig}`;
    it(testTitle, () => {
      const lr = 2;
      const optimizer = fromConfig ? new SGD({lr}) : new SGD(train.sgd(lr));
      const [x, y] = createVariables();
      const lossFn = () => {
        return K.add(x.read(), y.read()) as Scalar;
      };
      optimizer.updateVariables(lossFn, [x, y]);
      expectTensorsClose(x.read(), scalar(-2));
      expectTensorsClose(y.read(), scalar(-1));
    });
  }

  // TODO(cais): Test decay and momentum once they are supported by core
  //   SGDOptimizer.
});

// TODO(cais): The current version of deeplearn.js (0.4.2) seems to have a bug
//   in the multiplication and broadcasting on CPU. Change the following to
//   describeMathCPUAndGPU() when upgrading to a newer version with the bug
//   fixed.
describeMathGPU('Adam optimizer', () => {
  // The reference values of the values of kernel and bias used below are
  // obtained from PyKeras version 2.1.2 with tensorflow backend version
  // 1.6.0-dev20180130 (CPU only).
  // ```python
  // import keras
  // import numpy as np
  // batch_size = 2
  // input_size = 4
  // inputs = keras.layers.Input(shape=[input_size])
  // dense_layer = keras.layers.Dense(
  //     units=1, kernel_initializer='ones', bias_initializer='ones')
  // outputs = dense_layer(inputs)
  // model = keras.Model(inputs, outputs)
  // optimizer = keras.optimizers.Adam(lr=0.2)
  // model.compile(optimizer=optimizer, loss='mean_squared_error')
  // x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
  // y = np.array([[-10], [-20]])
  // for i in xrange(4):
  //   model.fit(x, y, batch_size=batch_size, epochs=1)
  //   print(dense_layer.get_weights())
  // ```
  const fromConfigValues = [false, true];
  for (const fromConfig of fromConfigValues) {
    const testTitle = `Update a Dense model: fromConfig=${fromConfig}`;
    it(testTitle, () => {
      const batchSize = 2;
      const inputSize = 4;

      const denseLayer = new Dense(
          {units: 1, kernelInitializer: 'Ones', biasInitializer: 'Ones'});
      const lr = 0.2;
      const beta1 = 0.9;
      const beta2 = 0.999;
      const adam = fromConfig ? new Adam({lr, beta1, beta2}) :
                                new Adam(train.adam(lr, beta1, beta2));

      const x = tensor2d([[1, 2, 3, 4], [5, 6, 7, 8]], [batchSize, inputSize]);
      const y = tensor2d([[-10], [-20]], [batchSize, 1]);
      denseLayer.apply(x);  // Call apply once to build the layer first.
      const lossFn = () => {
        return K.mean(metrics.mse(y, denseLayer.apply(x) as Tensor)) as Scalar;
      };
      for (let i = 0; i < 4; ++i) {
        adam.updateVariables(lossFn, denseLayer.trainableWeights);
        const weights = denseLayer.getWeights();
        const kernel = weights[0];
        const bias = weights[1];
        if (i === 0) {
          expectTensorsClose(kernel, tensor2d([0.8, 0.8, 0.8, 0.8], [4, 1]));
          expectTensorsClose(bias, tensor1d([0.8]));
        } else if (i === 1) {
          expectTensorsClose(
              kernel,
              tensor2d(
                  [0.60099494, 0.60098886, 0.60098487, 0.60098219], [4, 1]));
          expectTensorsClose(bias, tensor1d([0.60096622]));
        } else if (i === 2) {
          expectTensorsClose(
              kernel,
              tensor2d(
                  [0.40385836, 0.40383369, 0.40381753, 0.40380627], [4, 1]));
          expectTensorsClose(bias, tensor1d([0.40374136]));
        } else if (i === 3) {
          expectTensorsClose(
              kernel,
              tensor2d(
                  [0.20966624, 0.20960115, 0.20955873, 0.20952894], [4, 1]));
          expectTensorsClose(bias, tensor1d([0.20935827]));
        }
      }
    });
  }

  it('Create and use two instances of Adam', () => {
    const batchSize = 2;
    const inputSize = 4;

    const denseLayer = new Dense(
        {units: 1, kernelInitializer: 'Ones', biasInitializer: 'Ones'});
    const adam1 = new Adam({lr: 0.2});
    const adam2 = new Adam({lr: 0.2});

    const x = tensor2d([[1, 2, 3, 4], [5, 6, 7, 8]], [batchSize, inputSize]);
    const y = tensor2d([[-10], [-20]], [batchSize, 1]);
    denseLayer.apply(x);  // Call apply once to build the layer first.
    const lossFn = () => {
      return K.mean(metrics.mse(y, denseLayer.apply(x) as Tensor)) as Scalar;
    };
    expect(() => adam1.updateVariables(lossFn, denseLayer.trainableWeights))
        .not.toThrowError();
    expect(() => adam2.updateVariables(lossFn, denseLayer.trainableWeights))
        .not.toThrowError();
  });
});

// TODO(cais): The current version of deeplearn.js (0.4.2) seems to have a bug
//   in the multiplication and broadcasting on CPU. Change the following to
//   describeMathCPUAndGPU() when upgrading to a newer version with the bug
//   fixed.
describeMathGPU('RMSProp optimizer', () => {
  // The reference values of the values of kernel and bias used below are
  // obtained from PyKeras version 2.1.2 with tensorflow backend version
  // 1.5.0.
  // ```python
  // import keras
  // import numpy as np
  // batch_size = 2
  // input_size = 4
  // inputs = keras.layers.Input(shape=[input_size])
  // dense_layer = keras.layers.Dense(
  //     units=1, kernel_initializer='ones', bias_initializer='ones')
  // outputs = dense_layer(inputs)
  // model = keras.Model(inputs, outputs)
  // optimizer = keras.optimizers.rmsprop(lr=0.2)
  // model.compile(optimizer=optimizer, loss='mean_squared_error')
  // x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
  // y = np.array([[-10], [-20]])
  // for i in xrange(4):
  //   model.fit(x, y, batch_size=batch_size, epochs=1)
  //   print(dense_layer.get_weights())
  // ```
  const fromConfigValues = [false, true];
  for (const fromConfig of fromConfigValues) {
    const testTitle = `Update a Dense model: fromConfig=${fromConfig}`;
    it(testTitle, () => {
      const batchSize = 2;
      const inputSize = 4;

      const denseLayer = new Dense(
          {units: 1, kernelInitializer: 'Ones', biasInitializer: 'Ones'});
      const lr = 0.2;
      const rmsProp =
          fromConfig ? new RMSProp({lr}) : new RMSProp(train.rmsprop(lr));

      const x = tensor2d([[1, 2, 3, 4], [5, 6, 7, 8]], [batchSize, inputSize]);
      const y = tensor2d([[-10], [-20]], [batchSize, 1]);
      denseLayer.apply(x);  // Build the layer
      const lossFn = () => {
        return K.mean(metrics.mse(y, denseLayer.apply(x) as Tensor)) as Scalar;
      };
      for (let i = 0; i < 4; i++) {
        rmsProp.updateVariables(lossFn, denseLayer.trainableWeights);

        const weights = denseLayer.getWeights();
        const kernel = weights[0];
        const bias = weights[1];

        if (i === 0) {
          expectTensorsClose(
              kernel,
              tensor2d([0.3675446, 0.36754447, 0.36754453, 0.3675446], [4, 1]));
          expectTensorsClose(bias, tensor1d([0.36754453]));
        } else if (i === 1) {
          expectTensorsClose(
              kernel,
              tensor2d(
                  [0.01410502, 0.01352578, 0.01314825, 0.0128828], [4, 1]));
          expectTensorsClose(bias, tensor1d([0.01135707]));
        } else if (i === 2) {
          expectTensorsClose(
              kernel,
              tensor2d(
                  [-0.22224085, -0.22377077, -0.22476728, -0.22546779],
                  [4, 1]));
          expectTensorsClose(bias, tensor1d([-0.22948699]));
        } else if (i === 3) {
          expectTensorsClose(
              kernel,
              tensor2d(
                  [-0.3880985, -0.3909135, -0.3927458, -0.3940333], [4, 1]));
          expectTensorsClose(bias, tensor1d([-0.4014105]));
        }
      }
    });
  }
});

describeMathGPU('Adagrad optimizer', () => {
  // The reference values of the values of kernel and bias used below are
  // obtained from PyKeras version 2.1.2 with tensorflow backend version
  // 1.5.0.
  // ```python
  // import keras
  // import numpy as np
  // batch_size = 2
  // input_size = 4
  // inputs = keras.layers.Input(shape=[input_size])
  // dense_layer = keras.layers.Dense(
  //     units=1, kernel_initializer='ones', bias_initializer='ones')
  // outputs = dense_layer(inputs)
  // model = keras.Model(inputs, outputs)
  // optimizer = keras.optimizers.adagrad(lr=0.2)
  // model.compile(optimizer=optimizer, loss='mean_squared_error')
  // x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
  // y = np.array([[-10], [-20]])
  // for i in xrange(4):
  //   model.fit(x, y, batch_size=batch_size, epochs=1)
  //   print(dense_layer.get_weights())
  // ```
  const fromConfigValues = [false, true];
  for (const fromConfig of fromConfigValues) {
    const testTitle = `Update a Dense model: fromConfig=${fromConfig}`;
    it(testTitle, () => {
      const batchSize = 2;
      const inputSize = 4;

      const denseLayer = new Dense(
          {units: 1, kernelInitializer: 'Ones', biasInitializer: 'Ones'});
      const lr = 0.2;
      const adagrad =
          fromConfig ? new Adagrad({lr}) : new Adagrad(train.adagrad(lr));

      const x = tensor2d([[1, 2, 3, 4], [5, 6, 7, 8]], [batchSize, inputSize]);
      const y = tensor2d([[-10], [-20]], [batchSize, 1]);
      denseLayer.apply(x);  // Build the layer
      const lossFn = () => {
        return K.mean(metrics.mse(y, denseLayer.apply(x) as Tensor)) as Scalar;
      };
      for (let i = 0; i < 4; i++) {
        adagrad.updateVariables(lossFn, denseLayer.trainableWeights);

        const weights = denseLayer.getWeights();
        const kernel = weights[0];
        const bias = weights[1];

        if (i === 0) {
          expectTensorsClose(kernel, tensor2d([0.8, 0.8, 0.8, 0.8], [4, 1]));
          expectTensorsClose(bias, tensor1d([0.8]));
        } else if (i === 1) {
          expectTensorsClose(
              kernel,
              tensor2d([0.66737425, 0.6673338, 0.6673074, 0.66728884], [4, 1]));
          expectTensorsClose(bias, tensor1d([0.66718185]));
        } else if (i === 2) {
          expectTensorsClose(
              kernel,
              tensor2d([0.5636605, 0.56356317, 0.5634997, 0.56345505], [4, 1]));
          expectTensorsClose(bias, tensor1d([0.56319785]));
        } else if (i === 3) {
          expectTensorsClose(
              kernel,
              tensor2d([0.4770143, 0.4768495, 0.47674206, 0.47666645], [4, 1]));
          expectTensorsClose(bias, tensor1d([0.47623128]));
        }
      }
    });
  }
});


describeMathCPU('Optimizer-get', () => {
  for (const optimizerName
           of ['SGD', 'sgd', 'Adam', 'adam', 'RMSProp', 'rmsprop', 'Adagrad',
               'adagrad']) {
    it(`can instantiate ${optimizerName}`, () => {
      // tslint:disable-next-line:no-unused-expression
      new (getOptimizer(optimizerName))({});
    });
  }
});

// TODO(cais): Test the serializatoin and deserialization of optimizer objects.
