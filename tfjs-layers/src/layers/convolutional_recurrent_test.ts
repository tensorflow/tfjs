
/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';
import {Tensor} from '@tensorflow/tfjs-core';

import {serializeActivation, Tanh} from '../activations';
import {NonNeg, serializeConstraint, UnitNorm} from '../constraints';
import {sequential} from '../exports';
import * as tfl from '../index';
import {GlorotUniform, HeUniform, Ones, serializeInitializer} from '../initializers';
import {DataFormat, PaddingMode} from '../keras_format/common';
import {modelFromJSON} from '../models';
import {L1L2, serializeRegularizer} from '../regularizers';
import {getCartesianProductOfValues} from '../utils/generic_utils';
import {convertPythonicToTs, convertTsToPythonic} from '../utils/serialization_utils';
import {describeMathCPU, describeMathCPUAndGPU, describeMathCPUAndWebGL2, expectTensorsClose} from '../utils/test_utils';

import {ConvLSTM2DArgs, ConvLSTM2DCellArgs} from './convolutional_recurrent';

describeMathCPUAndGPU('ConvLSTM2DCell', () => {
  /**
   * The tensor values (output, h, c) can be obtained by the following Python
   * code
   *
   * sequence_len = 1
   * data_size = 8
   * data_channel = 3
   *
   * data_format = "channels_first"
   * filters = 9
   * kernel_size = 5
   * padding = "same"
   *
   * inputs = np.ones([1, sequence_len, data_channel, data_size, data_size])
   *
   * x = keras.Input(batch_shape=inputs.shape)
   *
   * kwargs = {'data_format': data_format,
   *           'return_state': True,
   *           'filters': filters,
   *           'kernel_size': kernel_size,
   *           'padding': padding}
   *
   * layer = keras.layers.ConvLSTM2D(kernel_initializer='ones',
   * bias_initializer="ones", recurrent_initializer='ones', **kwargs)
   * layer.build(inputs.shape)
   *
   * outputs = layer(x)
   *
   * model = keras.models.Model(x, outputs)
   *
   * y = model.predict(inputs)
   *
   * y[0].mean(), y[1].mean(), y[2].mean()
   */
  describe('should return the correct outputs', () => {
    const sequenceLength = 1;
    const inputSize = 8;
    const channels = 3;

    const dataFormatValues: DataFormat[] = ['channelsFirst', 'channelsLast'];
    const filterValues = [3, 5, 9];
    const kernelSizeValues = [3, 5];
    const paddingValues: PaddingMode[] = ['valid', 'same'];

    const testArgs = getCartesianProductOfValues(
                         dataFormatValues,
                         filterValues,
                         kernelSizeValues,
                         paddingValues,
                         ) as Array<[DataFormat, number, number, PaddingMode]>;

    for (const [dataFormat, filters, kernelSize, padding] of testArgs) {
      const testTitle = `for dataFormat=${dataFormat}, filters=${
          filters}, kernelSize=${kernelSize}, padding=${padding}`;

      it(testTitle, () => {
        const inputShape = dataFormat === 'channelsFirst' ?
            [sequenceLength, channels, inputSize, inputSize] :
            [sequenceLength, inputSize, inputSize, channels];

        const x = tfc.ones(inputShape);

        const cell = tfl.layers.convLstm2dCell({
          dataFormat,
          filters,
          kernelSize,
          padding,
          kernelInitializer: 'ones',
          recurrentInitializer: 'ones',
          biasInitializer: 'ones',
        });

        cell.build(x.shape);

        const outputSize =
            padding === 'same' ? inputSize : (inputSize - kernelSize + 1);

        const outShape = dataFormat === 'channelsFirst' ?
            [sequenceLength, filters, outputSize, outputSize] :
            [sequenceLength, outputSize, outputSize, filters];

        const initialH = tfc.zeros(outShape);

        const initialC = tfc.zeros(outShape);

        const [o, h, c] = cell.call([x, initialH, initialC], {});

        expect(o.shape).toEqual(outShape);
        expect(h.shape).toEqual(outShape);
        expect(c.shape).toEqual(outShape);

        expectTensorsClose(o.mean().flatten(), tfc.tensor1d([0.7615942]));
        expectTensorsClose(h.mean().flatten(), tfc.tensor1d([0.7615942]));
        expectTensorsClose(c.mean().flatten(), tfc.tensor1d([1]));
      });
    }
  });
});

describeMathCPU('ConvLSTM2D Symbolic', () => {
  const dtype = 'float32';

  const batchSize = 8;
  const sequenceLength = 10;
  const inputSize = 8;
  const channels = 3;

  const inputShape = [
    batchSize,
    sequenceLength,
    inputSize,
    inputSize,
    channels,
  ];

  const filters = 5;
  const kernelSize = [3, 3];

  const outputSize = inputSize - kernelSize[0] + 1;

  describe('should return the correct output shape', () => {
    it('for returnSequences=false, returnState=false', () => {
      const input =
          new tfl.SymbolicTensor('float32', inputShape, null, [], null);
      const convLstm = tfl.layers.convLstm2d({filters, kernelSize});
      const output = convLstm.apply(input) as tfl.SymbolicTensor;
      expect(output.shape).toEqual([8, 6, 6, 5]);
    });

    it('for returnSequences=false, returnState=true', () => {
      const returnState = true;

      const input = new tfl.SymbolicTensor(dtype, inputShape, null, [], null);

      const convLstm =
          tfl.layers.convLstm2d({filters, kernelSize, returnState});

      const output = convLstm.apply(input) as tfl.SymbolicTensor[];

      const outputShape = [batchSize, outputSize, outputSize, filters];

      expect(output.length).toEqual(3);
      expect(output[0].shape).toEqual(outputShape);
      expect(output[1].shape).toEqual(outputShape);
      expect(output[2].shape).toEqual(outputShape);
    });

    it('for returnSequences=true, returnState=false', () => {
      const returnSequences = true;

      const input = new tfl.SymbolicTensor(dtype, inputShape, null, [], null);

      const convLstm =
          tfl.layers.convLstm2d({filters, kernelSize, returnSequences});

      const output = convLstm.apply(input) as tfl.SymbolicTensor;

      const outputShape =
          [batchSize, sequenceLength, outputSize, outputSize, filters];

      expect(output.shape).toEqual(outputShape);
    });

    it('for returnSequences=true, returnState=true', () => {
      const returnSequences = true;
      const returnState = true;

      const input = new tfl.SymbolicTensor(dtype, inputShape, null, [], null);

      const convLstm = tfl.layers.convLstm2d(
          {filters, kernelSize, returnSequences, returnState});

      const output = convLstm.apply(input) as tfl.SymbolicTensor[];

      const outputShape =
          [batchSize, sequenceLength, outputSize, outputSize, filters];

      const stateShape = [batchSize, outputSize, outputSize, filters];

      expect(output.length).toEqual(3);
      expect(output[0].shape).toEqual(outputShape);
      expect(output[1].shape).toEqual(stateShape);
      expect(output[2].shape).toEqual(stateShape);
    });
  });

  it('should contain the correct number of weights', () => {
    const input = new tfl.SymbolicTensor(dtype, inputShape, null, [], null);

    const convLstm = tfl.layers.convLstm2d(
        {filters, kernelSize, returnSequences: true, returnState: true});

    convLstm.apply(input);

    expect(convLstm.trainable).toEqual(true);
    expect(convLstm.trainableWeights.length).toEqual(3);
    expect(convLstm.nonTrainableWeights.length).toEqual(0);
    expect(convLstm.weights.length).toEqual(3);
  });

  describe('should build the correct layer from exported config', () => {
    for (const implementation of [1, 2]) {
      it(`for implementation=${implementation}`, () => {
        const layer = tfl.layers.convLstm2d({
          filters,
          kernelSize,
          inputShape: inputShape.slice(1),
          implementation,
          padding: 'same',
          returnSequences: true,
        });

        const pythonicConfig = convertTsToPythonic(layer.getConfig());

        const tsConfig = convertPythonicToTs(pythonicConfig);

        const layerPrime =
            tfl.layers.convLstm2d(tsConfig as unknown as ConvLSTM2DArgs);

        expect(layerPrime.getConfig().filters).toEqual(filters);
        expect(layerPrime.getConfig().kernelSize).toEqual(kernelSize);
        expect(layerPrime.getConfig().implementation).toEqual(implementation);
      });
    }
  });

  describe('should return equal outputs with loaded model', () => {
    it('for simple model', async () => {
      const model = tfl.sequential();

      const layer = tfl.layers.convLstm2d({
        filters,
        kernelSize,
        padding: 'same',
        inputShape: inputShape.slice(1),
        returnSequences: true,
      });

      model.add(layer);

      const x = tfc.randomNormal(inputShape);

      const y = model.predict(x) as tfc.Tensor;

      let savedArtifacts: tfc.io.ModelArtifacts;

      await model.save(tfc.io.withSaveHandler(async (artifacts) => {
        savedArtifacts = artifacts;
        return null;
      }));

      const loadedModel =
          await tfl.loadLayersModel(tfc.io.fromMemory(savedArtifacts));

      const yPrime = loadedModel.predict(x) as tfc.Tensor;

      expect(model.inputs[0].shape).toEqual(loadedModel.inputs[0].shape);
      expect(model.outputs[0].shape).toEqual(loadedModel.outputs[0].shape);
      expectTensorsClose(yPrime, y);
    });

    it('for more complex model', async () => {
      const model = tfl.sequential();
      model.add(tfl.layers.convLstm2d({
        filters,
        kernelSize,
        inputShape: inputShape.slice(1),
      }));
      model.add(tfl.layers.dropout({rate: 0.2}));
      model.add(tfl.layers.flatten());
      model.add(tfl.layers.dense({units: 256, activation: 'relu'}));
      model.add(tfl.layers.dropout({rate: 0.3}));
      model.add(tfl.layers.dense({units: 6, activation: 'softmax'}));

      const x = tfc.randomNormal(inputShape);

      const y = model.predict(x) as tfc.Tensor;

      let savedArtifacts: tfc.io.ModelArtifacts;

      await model.save(tfc.io.withSaveHandler(async (artifacts) => {
        savedArtifacts = artifacts;
        return null;
      }));

      const loadedModel =
          await tfl.loadLayersModel(tfc.io.fromMemory(savedArtifacts));

      const yPrime = loadedModel.predict(x) as tfc.Tensor;

      expect(model.inputs[0].shape).toEqual(loadedModel.inputs[0].shape);
      expect(model.outputs[0].shape).toEqual(loadedModel.outputs[0].shape);
      expectTensorsClose(yPrime, y);
    });
  });
});

describeMathCPUAndWebGL2('ConvLSTM2D Tensor', () => {
  const filters = 5;
  const kernelSize = 3;

  const batchSize = 4;
  const sequenceLength = 2;
  const inputSize = 5;
  const channels = 3;

  const inputShape =
      [batchSize, sequenceLength, inputSize, inputSize, channels];

  const outputSize = inputSize - kernelSize + 1;

  describe('should run as expected', () => {
    const dropoutValues = [0.0, 0.1];
    const recurrentDropoutValues = [0.0, 0.1];
    const trainingValues = [true, false];
    const implementationValues = [1, 2];

    const args = getCartesianProductOfValues(
                     dropoutValues,
                     recurrentDropoutValues,
                     trainingValues,
                     implementationValues,
                     ) as Array<[number, number, boolean, number]>;

    for (const [dropout, recurrentDropout, training, implementation] of args) {
      const testTitle = `for dropout=${dropout}, recurrentDropout=${
          recurrentDropout},implementation=${implementation}, training=${
          training}, implementation=${implementation}`;

      it(testTitle, () => {
        const dropoutFunc = jasmine.createSpy('dropout').and.callFake(
            (x: Tensor, level: number, noiseShape?: number[], seed?: number) =>
                tfc.tidy(() => tfc.dropout(x, level, noiseShape, seed)));

        const convLstm = tfl.layers.convLstm2d({
          filters,
          kernelSize,
          kernelInitializer: 'ones',
          recurrentInitializer: 'ones',
          biasInitializer: 'ones',
          dropout,
          recurrentDropout,
          implementation,
          dropoutFunc,
        });

        const input = tfc.ones(inputShape);

        let dropoutCall = 0;
        if (dropout !== 0.0 && training) {
          dropoutCall += 4;
        }
        if (recurrentDropout !== 0.0 && training) {
          dropoutCall += 4;
        }

        let numTensors = 0;

        for (let i = 0; i < 2; i++) {
          tfc.dispose(convLstm.apply(input, {training}) as tfc.Tensor);

          expect(dropoutFunc).toHaveBeenCalledTimes((i + 1) * dropoutCall);

          if (i === 0) {
            numTensors = tfc.memory().numTensors;
          } else {
            expect(tfc.memory().numTensors).toEqual(numTensors);
          }
        }
      });
    }

    it('for stateful forward');
  });

  describe('should return the correct outputs', () => {
    const returnStateValues = [true, false];
    const returnSequencesValues = [true, false];
    const implementationValues = [1, 2];

    const args = getCartesianProductOfValues(
                     returnStateValues,
                     returnSequencesValues,
                     implementationValues,
                     ) as Array<[boolean, boolean, number]>;

    for (const [returnState, returnSequences, implementation] of args) {
      const testTitle = `for returnState=${returnState}, returnSequences=${
          returnSequences}, implementation=${implementation}`;

      it(testTitle, () => {
        const convLstm = tfl.layers.convLstm2d({
          filters,
          kernelSize,
          returnState,
          returnSequences,
          implementation,
          kernelInitializer: 'ones',
          recurrentInitializer: 'ones',
          biasInitializer: 'ones',
        });

        const input = tfc.ones(inputShape);

        let output = convLstm.apply(input);

        const expectedOutputValueAtT0 = 0.76159424;
        const expectedOutputValueAtT1 = 0.96402746;
        const expectedH = expectedOutputValueAtT1;
        const expectedC = 2.0;

        let expectedOutput: tfc.Tensor;
        if (returnSequences) {
          const outputAtT0 = tfc.mul(
              tfc.scalar(expectedOutputValueAtT0),
              tfc.ones([batchSize, 1, outputSize, outputSize, filters]));

          const outputAtT1 = tfc.mul(
              tfc.scalar(expectedOutputValueAtT1),
              tfc.ones([batchSize, 1, outputSize, outputSize, filters]));

          expectedOutput = tfc.concat([outputAtT0, outputAtT1], 1);
        } else {
          expectedOutput = tfc.mul(
              tfc.scalar(expectedOutputValueAtT1),
              tfc.ones([batchSize, outputSize, outputSize, filters]));
        }

        if (returnState) {
          output = output as tfc.Tensor[];

          expect(output.length).toEqual(3);

          expectTensorsClose(output[0], expectedOutput);

          expectTensorsClose(
              output[1],
              tfc.mul(
                  tfc.scalar(expectedH),
                  tfc.ones([batchSize, outputSize, outputSize, filters])));
          expectTensorsClose(
              output[2],
              tfc.mul(
                  tfc.scalar(expectedC),
                  tfc.ones([batchSize, outputSize, outputSize, filters])));
        } else {
          output = output as tfc.Tensor;

          expectTensorsClose(output, expectedOutput);
        }
      });
    }

    it('for nested model', () => {
      const model = tfl.sequential();

      const nestedModel = tfl.sequential();

      nestedModel.add(tfl.layers.convLstm2d({
        filters,
        kernelSize,
        kernelInitializer: 'ones',
        recurrentInitializer: 'ones',
        biasInitializer: 'ones',
        batchInputShape: inputShape
      }));

      model.add(nestedModel);

      const input = tfc.ones(inputShape);

      const output = model.apply(input) as tfc.Tensor;

      const expectedOutput = tfc.mul(
          tfc.scalar(0.96402746),
          tfc.ones([batchSize, outputSize, outputSize, filters]));

      expectTensorsClose(output, expectedOutput);
    });
  });
});

// TODO: Add GPU test once Gather supports 5 rank tensor.
describeMathCPU('should run BPTT correctly', () => {
  const filters = 3;
  const kernelSize = 3;
  const padding = 'same';

  const batchSize = 4;
  const sequenceLength = 2;
  const inputSize = 5;
  const channels = 1;

  const inputShape =
      [batchSize, sequenceLength, inputSize, inputSize, channels];

  /**
   * batch_size = 4
   * sequence_len = 2
   * data_size = 5
   * data_channel = 1
   *
   * filters = 3
   * kernel_size = 3
   * padding = "same"
   *
   * kwargs = {'filters': filters, 'kernel_size': kernel_size, 'padding':
   * padding, 'batch_input_shape': [batch_size, sequence_len, data_size,
   * data_size, data_channel], 'stateful': True}
   *
   * model = keras.Sequential()
   *
   * model.add(keras.layers.ConvLSTM2D(kernel_initializer='ones',
   * bias_initializer="ones", recurrent_initializer='ones', **kwargs))
   *
   * model.add(keras.layers.Flatten())
   *
   * model.add(keras.layers.Dense(units=1, kernel_initializer='zeros',
   * use_bias=False))
   *
   * model.compile(loss='mean_squared_error', optimizer='sgd')
   *
   * xs_1 = np.ones([batch_size, sequence_len, data_size, data_size,
   * data_channel])
   * xs_2 = np.zeros([batch_size, sequence_len, data_size, data_size,
   * data_channel])
   * xs = np.concatenate([xs_1, xs_2], 0)
   *
   * ys = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])
   *
   * history = model.fit(xs, ys, batch_size=batch_size, shuffle=False, epochs=3)
   * print(history.history)
   */
  it('for stateful BPTT', async () => {
    const model = tfl.sequential();

    model.add(tfl.layers.convLstm2d({
      filters,
      kernelSize,
      padding,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      biasInitializer: 'ones',
      stateful: true,
      batchInputShape: inputShape
    }));

    model.add(tfl.layers.flatten());

    model.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'zeros', useBias: false}));

    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const input = tfc.concat([tfc.ones(inputShape), tfc.zeros(inputShape)], 0);

    const output = tfc.tensor([1, 1, 1, 1, 0, 0, 0, 0]);

    const history =
        await model.fit(input, output, {batchSize, shuffle: false, epochs: 3});

    expect(history.history.loss.length).toBe(3);
    expect(history.history.loss[0]).toBeCloseTo(1.5441135168075562);
    expect(history.history.loss[1]).toBeCloseTo(3.209195613861084);
    expect(history.history.loss[2]).toBeCloseTo(3.7930736541748047);
  });

  /**
   * batch_size = 4
   * sequence_len = 2
   * data_size = 5
   * data_channel = 1
   *
   * filters = 3
   * kernel_size = 3
   * padding = "same"
   *
   * kwargs = {'filters': filters, 'kernel_size': kernel_size, 'padding':
   * padding, 'batch_input_shape': [batch_size, sequence_len, data_size,
   * data_size, data_channel]}
   *
   * model = keras.Sequential()
   *
   * model.add(keras.layers.ConvLSTM2D(kernel_initializer='ones',
   * bias_initializer="ones", recurrent_initializer='ones', **kwargs))
   *
   * model.add(keras.layers.Flatten())
   *
   * model.add(keras.layers.Dense(units=1, kernel_initializer='zeros',
   * use_bias=False))
   *
   * model.compile(loss='mean_squared_error', optimizer='sgd')
   *
   * xs_1 = np.ones([batch_size, sequence_len, data_size, data_size,
   * data_channel])
   * xs_2 = np.zeros([batch_size, sequence_len, data_size,
   * data_size, data_channel])
   * xs = np.concatenate([xs_1, xs_2], 0)
   *
   * ys = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])
   *
   * history = model.fit(xs, ys, batch_size=batch_size, shuffle=False, epochs=3)
   * print(history.history)
   */
  it('for normal BPTT', async () => {
    const model = tfl.sequential();

    model.add(tfl.layers.convLstm2d({
      filters,
      kernelSize,
      padding,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      biasInitializer: 'ones',
      batchInputShape: inputShape
    }));

    model.add(tfl.layers.flatten());

    model.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'zeros', useBias: false}));

    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const input = tfc.concat([tfc.ones(inputShape), tfc.zeros(inputShape)], 0);

    const output = tfc.tensor([1, 1, 1, 1, 0, 0, 0, 0]);

    const history =
        await model.fit(input, output, {batchSize, shuffle: false, epochs: 3});

    expect(history.history.loss.length).toBe(3);
    expect(history.history.loss[0]).toBeCloseTo(1.367607831954956);
    expect(history.history.loss[1]).toBeCloseTo(1.9423290491104126);
    expect(history.history.loss[2]).toBeCloseTo(2.00433349609375);
  });

  /**
   * batch_size = 4
   * sequence_len = 2
   * data_size = 5
   * data_channel = 1
   *
   * filters = 3
   * kernel_size = 3
   * padding = "same"
   *
   * kwargs = {'filters': filters, 'kernel_size': kernel_size, 'padding':
   * padding, 'batch_input_shape': [batch_size, sequence_len, data_size,
   * data_size, data_channel]}
   *
   * model = keras.Sequential()
   *
   * model.add(keras.layers.ConvLSTM2D(kernel_initializer='ones',
   * bias_initializer="ones", recurrent_initializer='ones', **kwargs))
   *
   * model.add(keras.layers.Flatten())
   *
   * model.add(keras.layers.Dense(units=1, kernel_initializer='zeros',
   * use_bias=False))
   *
   * model.compile(loss='mean_squared_error', optimizer='sgd')
   *
   * xs_1 = np.ones([batch_size, sequence_len, data_size, data_size,
   * data_channel])
   * xs_2 = np.zeros([batch_size, sequence_len, data_size,
   * data_size, data_channel])
   * xs = np.concatenate([xs_1, xs_2], 0)
   *
   * ys = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])
   *
   * model.fit(xs, ys, batch_size=batch_size, shuffle=False, epochs=2)
   *
   * history = model.fit(xs, ys, batch_size=batch_size, shuffle=False, epochs=3)
   * print(history.history)
   */
  it('with no leak', async () => {
    const model = tfl.sequential();

    model.add(tfl.layers.convLstm2d({
      filters,
      kernelSize,
      padding,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      biasInitializer: 'ones',
      batchInputShape: inputShape
    }));

    model.add(tfl.layers.flatten());

    model.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'zeros', useBias: false}));

    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const input = tfc.concat([tfc.ones(inputShape), tfc.zeros(inputShape)], 0);

    const output = tfc.tensor([1, 1, 1, 1, 0, 0, 0, 0]);

    // Serves as burn-in call for subsequent tracking of memory leak.
    await model.fit(input, output, {epochs: 2, batchSize, shuffle: false});

    const numTensors0 = tfc.memory().numTensors;
    const history =
        await model.fit(input, output, {epochs: 3, batchSize, shuffle: false});
    const numTensors1 = tfc.memory().numTensors;

    // Assert no memory leak.
    expect(numTensors1).toEqual(numTensors0);
    expect(history.history.loss.length).toEqual(3);
    expect(history.history.loss[0]).toBeCloseTo(2.00433349609375);
    expect(history.history.loss[1]).toBeCloseTo(2.0098776817321777);
    expect(history.history.loss[2]).toBeCloseTo(2.0099055767059326);
  });
});

describeMathCPU('ConvLSTM2D Serialization and Deserialization', () => {
  const cellConfig: ConvLSTM2DCellArgs = {
    filters: 32,
    kernelSize: 3,
    dataFormat: 'channelsLast',
    dilationRate: 2,
    padding: 'same',
    strides: 2,
    activation: 'tanh',
    recurrentActivation: 'tanh',
    useBias: true,
    kernelInitializer: 'glorotUniform',
    recurrentInitializer: 'heUniform',
    biasInitializer: 'ones',
    kernelRegularizer: 'l1l2',
    recurrentRegularizer: 'l1l2',
    biasRegularizer: 'l1l2',
    kernelConstraint: 'unitNorm',
    recurrentConstraint: 'unitNorm',
    biasConstraint: 'nonNeg',
    dropout: 0.1,
    recurrentDropout: 0.2,
    name: 'cell_1',
    batchSize: 12,
    batchInputShape: [12, 8, 8],
    inputShape: [8, 8],
    dtype: 'int32',
    inputDType: 'int32',
    trainable: true,
    implementation: 1,
    unitForgetBias: true,
  };

  const expectedCellConfigPrime = {
    name: 'cell_1',
    trainable: true,
    batchInputShape: [12, 8, 8],
    dtype: 'int32',
    filters: 32,
    kernelSize: [3, 3],
    dataFormat: 'channelsLast',
    dilationRate: [2, 2],
    padding: 'same',
    strides: [2, 2],
    activation: serializeActivation(new Tanh()),
    recurrentActivation: serializeActivation(new Tanh()),
    useBias: true,
    kernelInitializer: serializeInitializer(new GlorotUniform()),
    recurrentInitializer: serializeInitializer(new HeUniform()),
    biasInitializer: serializeInitializer(new Ones()),
    kernelRegularizer: serializeRegularizer(new L1L2()),
    recurrentRegularizer: serializeRegularizer(new L1L2()),
    biasRegularizer: serializeRegularizer(new L1L2()),
    activityRegularizer: serializeRegularizer(null),
    kernelConstraint: serializeConstraint(new UnitNorm({})),
    recurrentConstraint: serializeConstraint(new UnitNorm({})),
    biasConstraint: serializeConstraint(new NonNeg()),
    implementation: 1,
    unitForgetBias: true,
  };

  describe('ConvLSTM2DCell.getConfig', () => {
    it('should return the expected values', () => {
      const cell = tfl.layers.convLstm2dCell(cellConfig);

      const {dropout, recurrentDropout, ...configPrime} = cell.getConfig();

      expect(configPrime).toEqual(expectedCellConfigPrime);
      expect(dropout).toBeCloseTo(0.1);
      expect(recurrentDropout).toBeCloseTo(0.2);
    });
  });

  describe('ConvLSTM2D.getConfig', () => {
    it('should return the expected values', () => {
      const config: ConvLSTM2DArgs = {
        ...cellConfig,
        name: 'layer_1',
        ...{
          returnSequences: true,
          returnState: true,
          stateful: true,
          unroll: false,
          goBackwards: true,
          inputDim: 8,
          inputLength: 8,
        } as Omit<ConvLSTM2DArgs, keyof ConvLSTM2DCellArgs>
      };

      const cell = tfl.layers.convLstm2d(config);

      const {dropout, recurrentDropout, ...configPrime} = cell.getConfig();

      expect(configPrime).toEqual({
        ...expectedCellConfigPrime,
        name: 'layer_1',
        returnSequences: true,
        returnState: true,
        stateful: true,
        unroll: false,
        goBackwards: true,
      });
      expect(dropout).toBeCloseTo(0.1);
      expect(recurrentDropout).toBeCloseTo(0.2);
    });
  });

  it('should return equal outputs before and after', async () => {
    const model = sequential();

    const batchSize = 8;
    const sequenceLength = 1;
    const inputSize = 8;
    const channels = 3;

    const filters = 5;
    const kernelSize = 3;

    const layer = tfl.layers.convLstm2d({
      filters,
      kernelSize,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      returnSequences: true,
      dataFormat: 'channelsFirst',
      inputShape: [sequenceLength, channels, inputSize, inputSize]
    });

    model.add(layer);

    const x =
        tfc.ones([batchSize, sequenceLength, channels, inputSize, inputSize]);

    const y = model.predict(x) as tfc.Tensor;

    const json = model.toJSON(null, false);

    const modelPrime = await modelFromJSON({modelTopology: json});

    const yPrime = modelPrime.predict(x) as tfc.Tensor;

    expectTensorsClose(yPrime, y);
  });
});
