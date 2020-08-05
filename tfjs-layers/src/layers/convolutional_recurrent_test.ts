import {ones, tensor1d, zeros} from '@tensorflow/tfjs-core';

import {DataFormat, PaddingMode} from '../keras_format/common';
import {getCartesianProductOfValues} from '../utils/generic_utils';
import {describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {ConvLSTM2DCell} from './convolutional_recurrent';

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
          biasInitializer: 'ones'
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
