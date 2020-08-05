import {ones, Tensor, tensor1d, zeros} from '@tensorflow/tfjs-core';

import {sequential} from '../exports';
import {DataFormat, PaddingMode} from '../keras_format/common';
import {modelFromJSON} from '../models';
import {getCartesianProductOfValues} from '../utils/generic_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {ConvLSTM2D, ConvLSTM2DCell} from './convolutional_recurrent';

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

      const testTitle = `with dataFormat=${dataFormat}, filters=${
          filters}, kernelSize=${kernelSize}, padding=${padding}`;

      it(testTitle, () => {
        const inputShape = dataFormat === 'channelsFirst' ?
            [sequenceLength, dataChannel, dataSize, dataSize] :
            [sequenceLength, dataSize, dataSize, dataChannel];

        const x = ones(inputShape);

        const cell = new ConvLSTM2DCell({
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

describeMathCPU('ConvLSTM2D Symbolic', () => {});

describeMathCPUAndGPU('ConvLSTM2D Tensor', () => {});

describeMathCPU('ConvLSTM2D Serialization and Deserialization', () => {
  it('should', async () => {
    const model = sequential();

    const layer = new ConvLSTM2D({
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
