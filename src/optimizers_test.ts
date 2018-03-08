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
import {Scalar, scalar, Tensor, tensor1d, tensor2d} from 'deeplearn';

import * as K from './backend/deeplearnjs_backend';
import {Dense} from './layers/core';
import * as metrics from './metrics';
import {Adam, get as getOptimizer, Optimizer, RMSProp, SGD} from './optimizers';
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

function createFakeLossFunction(valueToReturn: number): () => Scalar {
  return () => scalar(valueToReturn);
}

describeMathCPUAndGPU('Optimizer', () => {
  // Install a mock K.gradients function.
  beforeEach(() => {
    spyOn(K, 'gradients').and.callFake(mockGradients);
  });

  class DummyOptimizer extends Optimizer {
    updateVariables(lossFn: () => Scalar, params: LayerVariable[]): void {}
  }

  it('initializes its member variables.', () => {
    const clipnorm = 1;
    const clipvalue = 2;
    const optimizer = new DummyOptimizer({clipnorm, clipvalue});
    expect(optimizer.clipnorm).toEqual(clipnorm);
    expect(optimizer.clipvalue).toEqual(clipvalue);
  });

  it('calculates gradients with no clipnorm or clipvalue.', () => {
    const optimizer = new DummyOptimizer({});
    const paramValues = [tensor1d([1, 0]), tensor2d([[1, 2], [3, 4]], [2, 2])];
    const params = paramValues.map(x => new LayerVariable(x));
    // See the mocked K.gradients() above for an explanation of these expected
    // values.
    const expected = [tensor1d([1, 1]), tensor2d([[-1, -1], [-1, -1]], [2, 2])];
    const loss = 1;
    const results =
        optimizer.getGradients(createFakeLossFunction(loss), params);
    for (let i = 0; i < results.length; i++) {
      expectTensorsClose(results[i], expected[i]);
    }
  });

  it('calculates gradients with clipnorm applied.', () => {
    const clipnorm = 1;
    const optimizer = new DummyOptimizer({clipnorm});
    const paramValues = [tensor1d([1, 2]), tensor2d([[3], [4]], [2, 1])];
    const params = paramValues.map(a => new LayerVariable(a));
    const loss = 1;
    const results =
        optimizer.getGradients(createFakeLossFunction(loss), params);
    // K.gradients() are all 1 or -1. Gradients norm is 2:
    const gradientsNorm = Math.sqrt(1 + 1 + 1 + 1);
    // clipnorm is 1. So expected gradients are now 1 / 2:
    const expectedClippedNorm = 1 / gradientsNorm;
    const expected = [
      tensor1d([expectedClippedNorm, expectedClippedNorm]),
      tensor2d([[-expectedClippedNorm], [-expectedClippedNorm]], [2, 1])
    ];
    for (let i = 0; i < results.length; i++) {
      expectTensorsClose(results[i], expected[i]);
    }
  });

  it('calculates gradients with clipvalue applied.', () => {
    const clipvalue = 2;
    const optimizer = new DummyOptimizer({clipvalue});
    const paramValues = [tensor1d([1, 2]), tensor2d([[3], [4]], [2, 1])];
    const params = paramValues.map(a => new LayerVariable(a));
    const loss = 3;
    // K.gradients() are all 3 * ones or -3 * ones (see mocked K.gradients()
    // above for an explanation of gradients used for these tests).
    const results =
        optimizer.getGradients(createFakeLossFunction(loss), params);
    // clipvalue is 2. So expected gradients are now clipvalue * ones or
    // -clipvalue * ones;
    const expected = [
      tensor1d([clipvalue, clipvalue]),
      tensor2d([[-clipvalue], [-clipvalue]], [2, 1])
    ];
    for (let i = 0; i < results.length; i++) {
      expectTensorsClose(results[i], expected[i]);
    }
  });

  it('calculates gradients with clipnorm and clipvalue applied.', () => {
    const clipnorm = 1;
    const clipvalue = 0.25;
    const optimizer = new DummyOptimizer({clipnorm, clipvalue});
    const paramValues = [tensor1d([1, 2]), tensor2d([[3], [4]], [2, 1])];
    const params = paramValues.map(a => new LayerVariable(a));
    const loss = 1;
    const results =
        optimizer.getGradients(createFakeLossFunction(loss), params);
    // clipvalue is 0.25. So expected gradients are now clipvalue * ones or
    // -clipvalue * ones;
    const expected = [
      tensor1d([clipvalue, clipvalue]),
      tensor2d([[-clipvalue], [-clipvalue]], [2, 1])
    ];
    for (let i = 0; i < results.length; i++) {
      expectTensorsClose(results[i], expected[i]);
    }
  });
});

describeMathCPUAndGPU('SGD', () => {
  // Install a mock K.gradients function.
  beforeEach(() => {
    spyOn(K, 'gradients').and.callFake(mockGradients);
  });

  function createVariables(): LayerVariable[] {
    return [K.zerosVariable([3]), K.zerosVariable([2, 2])];
  }

  it('initializes its member variables to default values.', () => {
    const optimizer = new SGD({});
    expect(optimizer.iterations).toEqual(0);
    expect(optimizer.lr).toEqual(0.01);
    expect(optimizer.momentum).toEqual(0.0);
    expect(optimizer.decay).toEqual(0.0);
    expect(optimizer.nesterov).toEqual(false);
  });

  it('initializes its member variables if specified.', () => {
    const lr = 0.2;
    const momentum = 3;
    const decay = 4;
    const nesterov = true;
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

  it('updates variables with no decay or momentum.', () => {
    const lr = 2;
    const loss = 1;
    const lossFn = createFakeLossFunction(loss);
    const optimizer = new SGD({lr});
    const initialValues = createVariables();
    const variables = createVariables();
    const grads = mockGradients(lossFn, variables);
    optimizer.updateVariables(lossFn, variables);
    for (let i = 0; i < variables.length; i++) {
      const gradient = grads[i];
      const velocity = K.neg(K.scalarTimesArray(scalar(lr), gradient));
      const initialValue = initialValues[i];
      const expectedValue = K.add(initialValue.read(), velocity);
      const result = variables[i].read();
      expectTensorsClose(result, expectedValue);
    }
  });

  it('updates variables with decay.', () => {
    const lr = 2;
    const decay = 3;
    const loss = 1;
    const lossFn = createFakeLossFunction(loss);
    const optimizer = new SGD({lr, decay});
    const initialValues = createVariables();
    const variables = createVariables();
    const grads = mockGradients(lossFn, variables);
    // Apply updates twice to ensure the number of iterations is taken into
    // account.
    optimizer.updateVariables(lossFn, variables);
    optimizer.updateVariables(lossFn, variables);
    const firstLR = lr / (1 + decay * 1);
    const secondLR = lr / (1 + decay * 2);
    for (let i = 0; i < variables.length; i++) {
      const gradient = grads[i];
      // Expected gradients are p - (firstLR * g) - (secondLR * g) or:
      // p - g(firstLR + secondLR)
      const velocity =
          K.neg(K.scalarTimesArray(scalar(firstLR + secondLR), gradient));
      const initialValue = initialValues[i];
      const expectedValue = K.add(initialValue.read(), velocity);
      const result = variables[i].read();
      expectTensorsClose(result, expectedValue);
    }
  });

  it('doesn\'t include momentum the first time through.', () => {
    const lr = 2;
    const momentum = 3;
    const loss = 1;
    const lossFn = createFakeLossFunction(loss);
    const optimizer = new SGD({lr, momentum});
    const initialValues = createVariables();
    const variables = createVariables();
    const grads = mockGradients(lossFn, variables);
    optimizer.updateVariables(lossFn, variables);
    for (let i = 0; i < variables.length; i++) {
      const gradient = grads[i];
      // Expected gradients:
      //
      // First velocity (v1) is momentum * moment - lr * g:
      //   v1 = -lr * g (since moment == 0).
      //
      // Thus, expected value = p + v1
      const v1 = K.neg(K.scalarTimesArray(scalar(lr), gradient));
      const initialValue = initialValues[i];
      const expectedValue = K.add(initialValue.read(), v1);
      const result = variables[i].read();
      expectTensorsClose(result, expectedValue);
    }
  });

  it('updates variables with momentum.', () => {
    const lr = 2;
    const momentum = 3;
    const loss = 1;
    const lossFn = createFakeLossFunction(loss);
    const optimizer = new SGD({lr, momentum});
    const initialValues = createVariables();
    const variables = createVariables();
    const grads = mockGradients(lossFn, variables);
    // Apply update the twice to be able to test momentum.
    optimizer.updateVariables(lossFn, variables);
    optimizer.updateVariables(lossFn, variables);
    for (let i = 0; i < variables.length; i++) {
      const gradient = grads[i];
      // Expected gradients:
      //
      // First velocity (v1) is momentum * moment - lr * g:
      //   v1 = -lr * g (since moment == 0).
      //
      // The moment then is updated to v1.
      //
      // The second velocity (v2) is momentum * moment - lr * g:
      //   v2 = momentum * v1 + v1
      //
      // Thus, expected value = p + v1 + v2
      const v1 = K.neg(K.scalarTimesArray(scalar(lr), gradient));
      const v2 = K.add(K.scalarTimesArray(scalar(momentum), v1), v1);
      const totalUpdate = K.add(v1, v2);
      const initialValue = initialValues[i];
      const expectedValue = K.add(initialValue.read(), totalUpdate);
      const result = variables[i].read();
      expectTensorsClose(result, expectedValue);
    }
  });

  it('updates multiple sets of variables with momentum.', () => {
    const lr = 2;
    const momentum = 3;
    const loss = 1;
    const lossFn = createFakeLossFunction(loss);
    const optimizer = new SGD({lr, momentum});
    const initialValues = createVariables();
    const variableSet1 = createVariables();
    const variableSet2 = createVariables();
    const combinedVariables = variableSet1.concat(variableSet2);
    const grads = mockGradients(lossFn, variableSet1);
    const combinedVariablesGrads = mockGradients(lossFn, combinedVariables);

    // Update the original variables once, each.
    optimizer.updateVariables(lossFn, variableSet1);
    optimizer.updateVariables(lossFn, variableSet2);
    for (const varSet of [variableSet1, variableSet2]) {
      for (let i = 0; i < varSet.length; i++) {
        const gradient = grads[i];
        // First velocity (v1) is momentum * moment - lr * g:
        //   v1 = -lr * g (since moment == 0).
        //
        // expectedValue = p + v1
        const v1 = K.neg(K.scalarTimesArray(scalar(lr), gradient));
        const initialValue = initialValues[i];
        const expectedValue = K.add(initialValue.read(), v1);
        const result = varSet[i].read();
        expectTensorsClose(result, expectedValue);
      }
    }
    // Update the combined variables. This will update variables in variableSet1
    // and variableSet 2, but no momentum should be applied to this set of
    // variables.
    optimizer.updateVariables(lossFn, combinedVariables);
    for (let i = 0; i < combinedVariables.length; i++) {
      const gradient = combinedVariablesGrads[i];
      // Expected value is now p - 2 * lr * g for each
      const velocity = K.scalarTimesArray(
          scalar(2.0), K.neg(K.scalarTimesArray(scalar(lr), gradient)));
      const initialValue = initialValues[i % initialValues.length];
      const expectedValue = K.add(initialValue.read(), velocity);
      const result = combinedVariables[i].read();
      expectTensorsClose(result, expectedValue);
    }
    // Update the original set a second time. This time, momentum should be
    // factored in.
    optimizer.updateVariables(lossFn, variableSet1);
    for (let i = 0; i < variableSet1.length; i++) {
      const gradient = grads[i];
      // Expected gradients:
      //
      // First velocity (v1) is momentum * moment - lr * g:
      //   v1 = -lr * g (since moment == 0).
      //
      // The moment then is updated to v1, and our parameter equals:
      // p = p - lr * g
      //
      // Above, we then updated the variable through the combinedVariables. So
      // now we should have:
      //
      // p = p - 2 * (lr * g).
      // We'll call -2 * (lr * g) the previousUpdate.
      //
      // We then update variableSet1 one more time. This update gives us v2,
      // which is momentum * moment - lr * g:
      //   v2 = momentum * v1 - lr * g
      //
      // Thus, expected value = p + previousUpdate + v2

      const negLRXG = K.neg(K.scalarTimesArray(scalar(lr), gradient));
      const previousUpdate = K.scalarTimesArray(scalar(2.0), negLRXG);
      const v2 = K.add(K.scalarTimesArray(scalar(momentum), negLRXG), negLRXG);
      const totalUpdate = K.add(previousUpdate, v2);
      const initialValue = initialValues[i];
      const expectedValue = K.add(initialValue.read(), totalUpdate);
      const result = variableSet1[i].read();
      expectTensorsClose(result, expectedValue);
    }
  });

  it('updates variables with decay and momentum.', () => {
    const lr = 2;
    const decay = 3;
    const momentum = 4;
    const loss = 1;
    const lossFn = createFakeLossFunction(loss);
    const optimizer = new SGD({lr, decay, momentum});
    const initialValues = createVariables();
    const variables = createVariables();
    const grads = mockGradients(lossFn, variables);
    // Apply updates twice to be able to test decay and momentum.
    optimizer.updateVariables(lossFn, variables);
    optimizer.updateVariables(lossFn, variables);
    const lr1 = lr / (1 + decay * 1);
    const lr2 = lr / (1 + decay * 2);
    for (let i = 0; i < variables.length; i++) {
      const gradient = grads[i];
      // Expected gradients:
      //
      // lr1 = lr / (1 + decay * 1)
      // lr2 = lr / (1 + decay * 2)
      //
      // First velocity (v1) is momentum * moment - lr1 * g:
      //   v1 = -lr1 * g (since moment == 0).
      //
      // The moment then is updated to v1.
      //
      // The second velocity (v2) is momentum * moment - lr2 * g:
      //   v2 = momentum * v1 - lr2 * g
      //
      // Thus, expected value = p + v1 + v2
      const v1 = K.neg(K.scalarTimesArray(scalar(lr1), gradient));
      const v2 = K.subtract(
          K.scalarTimesArray(scalar(momentum), v1),
          K.scalarTimesArray(scalar(lr2), gradient));
      const totalUpdate = K.add(v1, v2);
      const initialValue = initialValues[i];
      const expectedValue = K.add(initialValue.read(), totalUpdate);
      const result = variables[i].read();
      expectTensorsClose(result, expectedValue);
    }
  });

  it('updates variables with decay, momentum, and nesterov.', () => {
    const lr = 2;
    const decay = 3;
    const momentum = 4;
    const nesterov = true;
    const loss = 1;
    const lossFn = createFakeLossFunction(loss);
    const optimizer = new SGD({lr, decay, momentum, nesterov});
    const initialValues = createVariables();
    const variables = createVariables();
    const grads = mockGradients(lossFn, variables);
    // Apply updates twice to be able to test decay and momentum.
    optimizer.updateVariables(lossFn, variables);
    optimizer.updateVariables(lossFn, variables);
    const lr1 = lr / (1 + decay * 1);
    const lr2 = lr / (1 + decay * 2);
    for (let i = 0; i < variables.length; i++) {
      const gradient = grads[i];
      // Expected gradients:
      //
      // lr1 = lr / (1 + decay * 1)
      // lr2 = lr / (1 + decay * 2)
      //
      // First velocity (v1) is momentum * moment - lr1 * g:
      //   v1 = -lr1 * g (since moment == 0).
      //   nesterov1 = momentum * v1 - lr1 * g
      //
      // The moment then is updated to v1.
      //
      // The second velocity (v2) is momentum * moment - lr2 * g:
      //   v2 = momentum * v1 - lr2 * g
      //   nesterov2 = momentum * v2 - lr2 * g
      //
      // Thus, expected value = p + nesterov1 + nesterov2
      const momentumScalar = scalar(momentum);
      const negLR1Xgradient = K.neg(K.scalarTimesArray(scalar(lr1), gradient));
      const v1 = negLR1Xgradient;
      const nesterov1 =
          K.add(K.scalarTimesArray(momentumScalar, v1), negLR1Xgradient);

      const negLR2Xg = K.neg(K.scalarTimesArray(scalar(lr2), gradient));
      const v2 = K.add(K.scalarTimesArray(scalar(momentum), v1), negLR2Xg);
      const nesterov2 = K.add(K.scalarTimesArray(momentumScalar, v2), negLR2Xg);
      const totalUpdate = K.add(nesterov1, nesterov2);
      const initialValue = initialValues[i];
      const expectedValue = K.add(initialValue.read(), totalUpdate);
      const result = variables[i].read();
      expectTensorsClose(result, expectedValue);
    }
  });
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
  it('Update a Dense model', () => {
    const batchSize = 2;
    const inputSize = 4;

    const denseLayer = new Dense(
        {units: 1, kernelInitializer: 'Ones', biasInitializer: 'Ones'});
    const adam = new Adam({lr: 0.2});
    // Check the name scope of the variables.
    expect(adam.lr.name.indexOf('Adam/lr')).toEqual(0);
    expect(adam.beta1.name.indexOf('Adam/beta_1')).toEqual(0);
    expect(adam.beta2.name.indexOf('Adam/beta_2')).toEqual(0);
    expect(adam.decay.name.indexOf('Adam/decay')).toEqual(0);

    const x = tensor2d([[1, 2, 3, 4], [5, 6, 7, 8]], [batchSize, inputSize]);
    const y = tensor2d([[-10], [-20]], [batchSize, 1]);
    denseLayer.apply(x);  // Call apply once to build the layer first.
    const lossFn = () => {
      return K.mean(metrics.mse(y, denseLayer.apply(x) as Tensor)) as Scalar;
    };
    for (let i = 0; i < 4; ++i) {
      adam.updateVariables(lossFn, denseLayer.trainableWeights);
      expectTensorsClose(adam.iterations.read(), scalar(i + 1, 'int32'));
      const weights = denseLayer.getWeights();
      const kernel = weights[0];
      const bias = weights[1];
      if (i === 0) {
        expectTensorsClose(kernel, tensor2d([0.8, 0.8, 0.8, 0.8], [4, 1]));
        expectTensorsClose(bias, tensor1d([0.8]));
      } else if (i === 1) {
        expectTensorsClose(
            kernel,
            tensor2d([0.60099494, 0.60098886, 0.60098487, 0.60098219], [4, 1]));
        expectTensorsClose(bias, tensor1d([0.60096622]));
      } else if (i === 2) {
        expectTensorsClose(
            kernel,
            tensor2d([0.40385836, 0.40383369, 0.40381753, 0.40380627], [4, 1]));
        expectTensorsClose(bias, tensor1d([0.40374136]));
      } else if (i === 3) {
        expectTensorsClose(
            kernel,
            tensor2d([0.20966624, 0.20960115, 0.20955873, 0.20952894], [4, 1]));
        expectTensorsClose(bias, tensor1d([0.20935827]));
      }
    }
  });

  it('Create and use two instances of Adam', () => {
    const batchSize = 2;
    const inputSize = 4;

    const denseLayer = new Dense(
        {units: 1, kernelInitializer: 'Ones', biasInitializer: 'Ones'});
    const adam1 = new Adam({lr: 0.2});
    const adam2 = new Adam({lr: 0.2});
    expect(adam1.lr.name.indexOf('Adam/lr')).toEqual(0);
    expect(adam1.beta1.name.indexOf('Adam/beta_1')).toEqual(0);
    expect(adam1.beta2.name.indexOf('Adam/beta_2')).toEqual(0);
    expect(adam1.decay.name.indexOf('Adam/decay')).toEqual(0);
    expect(adam2.lr.name.indexOf('Adam/lr')).toEqual(0);
    expect(adam2.beta1.name.indexOf('Adam/beta_1')).toEqual(0);
    expect(adam2.beta2.name.indexOf('Adam/beta_2')).toEqual(0);
    expect(adam2.decay.name.indexOf('Adam/decay')).toEqual(0);
    expect(adam1.lr.name).not.toEqual(adam2.lr.name);
    expect(adam1.beta1.name).not.toEqual(adam2.beta1.name);
    expect(adam1.beta2.name).not.toEqual(adam2.beta2.name);
    expect(adam1.decay.name).not.toEqual(adam2.decay.name);

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
  it('Update a Dense model', () => {
    const batchSize = 2;
    const inputSize = 4;

    const denseLayer = new Dense(
        {units: 1, kernelInitializer: 'Ones', biasInitializer: 'Ones'});
    const rmsProp = new RMSProp({lr: 0.2});

    const x = tensor2d([[1, 2, 3, 4], [5, 6, 7, 8]], [batchSize, inputSize]);
    const y = tensor2d([[-10], [-20]], [batchSize, 1]);
    denseLayer.apply(x);  // Build the layer
    const lossFn = () => {
      return K.mean(metrics.mse(y, denseLayer.apply(x) as Tensor)) as Scalar;
    };
    for (let i = 0; i < 4; i++) {
      rmsProp.updateVariables(lossFn, denseLayer.trainableWeights);
      expectTensorsClose(rmsProp.iterations.read(), scalar(i + 1, 'int32'));

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
            tensor2d([0.01410502, 0.01352578, 0.01314825, 0.0128828], [4, 1]));
        expectTensorsClose(bias, tensor1d([0.01135707]));
      } else if (i === 2) {
        expectTensorsClose(
            kernel,
            tensor2d(
                [-0.22224085, -0.22377077, -0.22476728, -0.22546779], [4, 1]));
        expectTensorsClose(bias, tensor1d([-0.22948699]));
      } else if (i === 3) {
        expectTensorsClose(
            kernel,
            tensor2d([-0.3880985, -0.3909135, -0.3927458, -0.3940333], [4, 1]));
        expectTensorsClose(bias, tensor1d([-0.4014105]));
      }
    }
  });
});

describeMathCPU('Optimizer get', () => {
  for (const optimizerName
           of ['SGD', 'sgd', 'Adam', 'adam', 'RMSProp', 'rmsprop']) {
    it(`can instantiate ${optimizerName}`, () => {
      // tslint:disable-next-line:no-unused-expression
      new (getOptimizer(optimizerName))({});
    });
  }
});

// TODO(cais): Test the serializatoin and deserialization of optimizer objects.
