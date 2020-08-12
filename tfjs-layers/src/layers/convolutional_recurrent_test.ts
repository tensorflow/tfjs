
import * as tfc from '@tensorflow/tfjs-core';

import {sequential} from '../exports';
import * as tfl from '../index';
import {DataFormat, PaddingMode} from '../keras_format/common';
import {modelFromJSON} from '../models';
import {getCartesianProductOfValues} from '../utils/generic_utils';
import {convertPythonicToTs, convertTsToPythonic} from '../utils/serialization_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {ConvLSTM2DArgs} from './convolutional_recurrent';

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
    const dataSize = 8;
    const dataChannel = 3;

    const dataFormatOptions: DataFormat[] = ['channelsFirst', 'channelsLast'];
    const filterOptions = [3, 5, 9];
    const kernelSizeOptions = [3, 5];
    const paddingOptions: PaddingMode[] = ['valid', 'same'];

    const testArgs =
        getCartesianProductOfValues(
            dataFormatOptions, filterOptions, kernelSizeOptions,
            paddingOptions) as Array<[DataFormat, number, number, PaddingMode]>;

    for (const [dataFormat, filters, kernelSize, padding] of testArgs) {
      const testTitle = `for dataFormat=${dataFormat}, filters=${
          filters}, kernelSize=${kernelSize}, padding=${padding}`;

      it(testTitle, () => {
        const inputShape = dataFormat === 'channelsFirst' ?
            [sequenceLength, dataChannel, dataSize, dataSize] :
            [sequenceLength, dataSize, dataSize, dataChannel];

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

        const outSize =
            padding === 'same' ? dataSize : (dataSize - kernelSize + 1);

        const outShape = dataFormat === 'channelsFirst' ?
            [sequenceLength, filters, outSize, outSize] :
            [sequenceLength, outSize, outSize, filters];

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
  describe('should return the correct output shape', () => {
    it('for returnSequences=false, returnState=false', () => {
      const input =
          new tfl.SymbolicTensor('float32', [8, 10, 8, 8, 3], null, [], null);
      const convLstm = tfl.layers.convLstm2d({filters: 5, kernelSize: 3});
      const output = convLstm.apply(input) as tfl.SymbolicTensor;
      expect(output.shape).toEqual([8, 6, 6, 5]);
    });

    it('for returnSequences=false, returnState=true', () => {
      const input =
          new tfl.SymbolicTensor('float32', [8, 10, 8, 8, 3], null, [], null);
      const convLstm =
          tfl.layers.convLstm2d({filters: 5, kernelSize: 3, returnState: true});
      const output = convLstm.apply(input) as tfl.SymbolicTensor[];
      expect(output.length).toEqual(3);
      expect(output[0].shape).toEqual([8, 6, 6, 5]);
      expect(output[1].shape).toEqual([8, 6, 6, 5]);
      expect(output[2].shape).toEqual([8, 6, 6, 5]);
    });

    it('for returnSequences=true, returnState=false', () => {
      const input =
          new tfl.SymbolicTensor('float32', [8, 10, 8, 8, 3], null, [], null);
      const convLstm = tfl.layers.convLstm2d(
          {filters: 5, kernelSize: 3, returnSequences: true});
      const output = convLstm.apply(input) as tfl.SymbolicTensor;
      expect(output.shape).toEqual([8, 10, 6, 6, 5]);
    });

    it('for returnSequences=true, returnState=true', () => {
      const input =
          new tfl.SymbolicTensor('float32', [8, 10, 8, 8, 3], null, [], null);
      const convLstm = tfl.layers.convLstm2d({
        filters: 5,
        kernelSize: 3,
        returnSequences: true,
        returnState: true
      });
      const output = convLstm.apply(input) as tfl.SymbolicTensor[];
      expect(output.length).toEqual(3);
      expect(output[0].shape).toEqual([8, 10, 6, 6, 5]);
      expect(output[1].shape).toEqual([8, 6, 6, 5]);
      expect(output[2].shape).toEqual([8, 6, 6, 5]);
    });
  });

  it('should contain the correct numbers of weights', () => {
    const input =
        new tfl.SymbolicTensor('float32', [8, 10, 8, 8, 3], null, [], null);
    const convLstm = tfl.layers.convLstm2d(
        {filters: 5, kernelSize: 3, returnSequences: true, returnState: true});
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
          filters: 5,
          kernelSize: 3,
          padding: 'same',
          returnSequences: true,
          inputShape: [10, 8, 8, 3],
          implementation,
        });

        const pythonicConfig = convertTsToPythonic(layer.getConfig());

        const tsConfig = convertPythonicToTs(pythonicConfig);

        const layerPrime =
            tfl.layers.convLstm2d(tsConfig as unknown as ConvLSTM2DArgs);

        expect(layerPrime.getConfig().filters).toEqual(5);
        expect(layerPrime.getConfig().implementation).toEqual(implementation);
      });
    }
  });

  describe('should return equal outputs with loaded model', () => {
    it('for simple model', async () => {
      const model = tfl.sequential();

      const layer = tfl.layers.convLstm2d({
        filters: 5,
        kernelSize: 3,
        padding: 'same',
        returnSequences: true,
        inputShape: [10, 8, 8, 3]
      });

      model.add(layer);

      const x = tfc.randomNormal([8, 10, 8, 8, 3]);
      const y = model.predict(x) as tfc.Tensor;

      let savedArtifacts: tfc.io.ModelArtifacts;

      await model.save(tfc.io.withSaveHandler(async (artifacts) => {
        savedArtifacts = artifacts;
        return null;
      }));

      const loadedModel =
          await tfl.loadLayersModel(tfc.io.fromMemory(savedArtifacts));

      expect(model.inputs[0].shape).toEqual(loadedModel.inputs[0].shape);
      expect(model.outputs[0].shape).toEqual(loadedModel.outputs[0].shape);
      expectTensorsClose(loadedModel.predict(x) as tfc.Tensor, y);
    });

    it('for more complex model', async () => {
      const model = tfl.sequential();

      const layer = tfl.layers.convLstm2d({
        filters: 64,
        kernelSize: 3,
        returnSequences: false,
        inputShape: [10, 8, 8, 3]
      });

      model.add(layer);
      model.add(tfl.layers.dropout({rate: 0.2}));
      model.add(tfl.layers.flatten());
      model.add(tfl.layers.dense({units: 256, activation: 'relu'}));
      model.add(tfl.layers.dropout({rate: 0.3}));
      model.add(tfl.layers.dense({units: 6, activation: 'softmax'}));

      const x = tfc.randomNormal([8, 10, 8, 8, 3]);
      const y = model.predict(x) as tfc.Tensor;

      let savedArtifacts: tfc.io.ModelArtifacts;

      await model.save(tfc.io.withSaveHandler(async (artifacts) => {
        savedArtifacts = artifacts;
        return null;
      }));

      const loadedModel =
          await tfl.loadLayersModel(tfc.io.fromMemory(savedArtifacts));

      expect(model.inputs[0].shape).toEqual(loadedModel.inputs[0].shape);
      expect(model.outputs[0].shape).toEqual(loadedModel.outputs[0].shape);
      expectTensorsClose(loadedModel.predict(x) as tfc.Tensor, y);
    });
  });
});

describeMathCPUAndGPU('ConvLSTM2D Tensor', () => {
  const filters = 5;
  const kernelSize = 3;

  const batchSize = 4;
  const sequence = 3;
  const inputSize = 5;
  const channels = 3;

  const dropouts = [0.0, 0.1];
  const recurrentDropouts = [0.0, 0.1];
  const trainings = [true, false];
  const implementations = [1, 2];
  const returnStates = [false, true];
  const returnSequencesValues = [false, true];

  const args = getCartesianProductOfValues(
                   dropouts, recurrentDropouts, trainings, implementations,
                   returnStates, returnSequencesValues) as
      Array<[number, number, boolean, number, boolean, boolean]>;

  for (const
           [dropout, recurrentDropout, training, implementation, returnState,
            returnSequences] of args) {
    const testTitle = `for dropout=${dropout}, recurrentDropout=${
        recurrentDropout},implementation=${implementation}, training=${
        training}, implementation=${implementation}, returnState=${
        returnState}, returnSequence=${returnSequences}`;

    it(testTitle, () => {
      const convLstm = tfl.layers.convLstm2d({
        filters,
        kernelSize,
        kernelInitializer: 'ones',
        recurrentInitializer: 'ones',
        biasInitializer: 'ones',
        dropout,
        recurrentDropout,
        implementation,
        returnState,
        returnSequences,
      });

      const input =
          tfc.ones([batchSize, sequence, inputSize, inputSize, channels]);

      spyOn(tfc, 'dropout').and.callThrough();
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

        expect(tfc.dropout).toHaveBeenCalledTimes((i + 1) * dropoutCall);

        if (i === 0) {
          numTensors = tfc.memory().numTensors;
        } else {
          expect(tfc.memory().numTensors).toEqual(numTensors);
        }
      }
    });
  }

  it('for forward stateful');

  it('with goBackwards=false');

  it('with goBackwards=true');

  it('with nested model');
});

describe('should run BPPT correctly', () => {
  it('for stateful BPPT');

  it('for normal BPPT');

  it('with no leak');
});

describeMathCPU('ConvLSTM2D Serialization and Deserialization', () => {
  it('should return equal outputs before and after', async () => {
    const model = sequential();

    const layer = tfl.layers.convLstm2d({
      filters: 5,
      kernelSize: 3,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      returnSequences: true,
      dataFormat: 'channelsFirst',
      inputShape: [1, 3, 8, 8]
    });

    model.add(layer);

    const x = tfc.ones([1, 1, 3, 8, 8]);
    const y = model.predict(x) as tfc.Tensor;

    const json = model.toJSON(null, false);

    const modelPrime = await modelFromJSON({modelTopology: json});

    const yPrime = modelPrime.predict(x) as tfc.Tensor;

    expectTensorsClose(yPrime, y);
  });
});
