import {io, ones, randomNormal, Tensor, tensor1d, zeros} from '@tensorflow/tfjs-core';

import {sequential} from '../exports';
import * as tfl from '../index';
import {DataFormat, PaddingMode} from '../keras_format/common';
import {modelFromJSON} from '../models';
import {getCartesianProductOfValues} from '../utils/generic_utils';
import {convertPythonicToTs, convertTsToPythonic} from '../utils/serialization_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {ConvLSTM2DArgs} from './convolutional_recurrent';

describeMathCPUAndGPU('ConvLSTM2DCell', () => {
  describe('should return the correct outputs', () => {
    const sequenceLength = 1;
    const dataSize = 8;
    const dataChannel = 3;

    const dataFormatOptions: DataFormat[] = ['channelsFirst', 'channelsLast'];
    const filterOptions = [3, 5, 9];
    const kernelSizeOptions = [3, 5];
    const paddingOptions: PaddingMode[] = ['valid', 'same'];

    const testArgs = getCartesianProductOfValues(
        dataFormatOptions,
        filterOptions,
        kernelSizeOptions,
        paddingOptions,
    );

    for (const args of testArgs) {
      const [dataFormat, filters, kernelSize, padding] =
          args as [DataFormat, number, number, PaddingMode];

      const testTitle = `for dataFormat=${dataFormat}, filters=${
          filters}, kernelSize=${kernelSize}, padding=${padding}`;

      it(testTitle, () => {
        const inputShape = dataFormat === 'channelsFirst' ?
            [sequenceLength, dataChannel, dataSize, dataSize] :
            [sequenceLength, dataSize, dataSize, dataChannel];

        const x = ones(inputShape);

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

        const initialH = zeros(outShape);

        const initialC = zeros(outShape);

        const [o, h, c] = cell.call([x, initialH, initialC], {});

        expect(o.shape).toEqual(outShape);
        expect(h.shape).toEqual(outShape);
        expect(c.shape).toEqual(outShape);

        expectTensorsClose(o.mean().flatten(), tensor1d([0.7615942]));
        expectTensorsClose(h.mean().flatten(), tensor1d([0.7615942]));
        expectTensorsClose(c.mean().flatten(), tensor1d([1]));
      });
    }
  });
});

describeMathCPU('ConvLSTM2D Symbolic', () => {
  describe('should return the correct output shape', () => {
    it('for returnSequences=false, returnState=false', () => {
      const input =
          new tfl.SymbolicTensor('float32', [8, 10, 8, 8, 3], null, [], null);
      const lstm = tfl.layers.convLstm2d({filters: 5, kernelSize: 3});
      const output = lstm.apply(input) as tfl.SymbolicTensor;
      expect(output.shape).toEqual([8, 6, 6, 5]);
    });

    it('for returnSequences=false, returnState=true', () => {
      const input =
          new tfl.SymbolicTensor('float32', [8, 10, 8, 8, 3], null, [], null);
      const lstm =
          tfl.layers.convLstm2d({filters: 5, kernelSize: 3, returnState: true});
      const output = lstm.apply(input) as tfl.SymbolicTensor[];
      expect(output.length).toEqual(3);
      expect(output[0].shape).toEqual([8, 6, 6, 5]);
      expect(output[1].shape).toEqual([8, 6, 6, 5]);
      expect(output[2].shape).toEqual([8, 6, 6, 5]);
    });

    it('for returnSequences=true, returnState=false', () => {
      const input =
          new tfl.SymbolicTensor('float32', [8, 10, 8, 8, 3], null, [], null);
      const lstm = tfl.layers.convLstm2d(
          {filters: 5, kernelSize: 3, returnSequences: true});
      const output = lstm.apply(input) as tfl.SymbolicTensor;
      expect(output.shape).toEqual([8, 10, 6, 6, 5]);
    });

    it('for returnSequences=true, returnState=true', () => {
      const input =
          new tfl.SymbolicTensor('float32', [8, 10, 8, 8, 3], null, [], null);
      const lstm = tfl.layers.convLstm2d({
        filters: 5,
        kernelSize: 3,
        returnSequences: true,
        returnState: true
      });
      const output = lstm.apply(input) as tfl.SymbolicTensor[];
      expect(output.length).toEqual(3);
      expect(output[0].shape).toEqual([8, 10, 6, 6, 5]);
      expect(output[1].shape).toEqual([8, 6, 6, 5]);
      expect(output[2].shape).toEqual([8, 6, 6, 5]);
    });
  });

  it('should contain the correct numbers of weights', () => {
    const input =
        new tfl.SymbolicTensor('float32', [8, 10, 8, 8, 3], null, [], null);
    const lstm = tfl.layers.convLstm2d(
        {filters: 5, kernelSize: 3, returnSequences: true, returnState: true});
    lstm.apply(input);
    expect(lstm.trainable).toEqual(true);
    expect(lstm.trainableWeights.length).toEqual(3);
    expect(lstm.nonTrainableWeights.length).toEqual(0);
    expect(lstm.weights.length).toEqual(3);
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

      const x = randomNormal([8, 10, 8, 8, 3]);
      const y = model.predict(x) as Tensor;

      let savedArtifacts: io.ModelArtifacts;

      await model.save(io.withSaveHandler(async (artifacts) => {
        savedArtifacts = artifacts;
        return null;
      }));

      const loadedModel =
          await tfl.loadLayersModel(io.fromMemory(savedArtifacts));

      expect(model.inputs[0].shape).toEqual(loadedModel.inputs[0].shape);
      expect(model.outputs[0].shape).toEqual(loadedModel.outputs[0].shape);
      expectTensorsClose(loadedModel.predict(x) as Tensor, y);
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

      const x = randomNormal([8, 10, 8, 8, 3]);
      const y = model.predict(x) as Tensor;

      let savedArtifacts: io.ModelArtifacts;

      await model.save(io.withSaveHandler(async (artifacts) => {
        savedArtifacts = artifacts;
        return null;
      }));

      const loadedModel =
          await tfl.loadLayersModel(io.fromMemory(savedArtifacts));

      expect(model.inputs[0].shape).toEqual(loadedModel.inputs[0].shape);
      expect(model.outputs[0].shape).toEqual(loadedModel.outputs[0].shape);
      expectTensorsClose(loadedModel.predict(x) as Tensor, y);
    });
  });
});

describeMathCPUAndGPU('ConvLSTM2D Tensor', () => {});

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

    const x = ones([1, 1, 3, 8, 8]);
    const y = model.predict(x) as Tensor;

    const json = model.toJSON(null, false);

    const modelPrime = await modelFromJSON({modelTopology: json});

    const yPrime = modelPrime.predict(x) as Tensor;

    expectTensorsClose(yPrime, y);
  });
});
