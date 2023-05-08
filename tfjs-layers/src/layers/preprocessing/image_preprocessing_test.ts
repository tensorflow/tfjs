/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { Tensor, randomNormal, mul, add} from '@tensorflow/tfjs-core';
import { Rescaling } from './image_preprocessing';
import { describeMathCPUAndGPU, expectTensorsClose } from '../../utils/test_utils';

describeMathCPUAndGPU('Rescaling Layer', () => {

  it('Check if input shape matches output shape', () => {
    const scale = 1.0 / 127.5;
    const offset = 0;
    const input = randomNormal([2, 4, 5, 3]);
    const expectedOutputTensor = add(mul(input, scale), offset);
    const scalingLayer = new Rescaling({scale, offset});
    const layerOutputTensor = scalingLayer.apply(input) as Tensor;
    expect(expectedOutputTensor.shape).toEqual(layerOutputTensor.shape);
  });

  it('Rescales input layer based on scaling factor and offset', () => {
    const scale = 1.0 / 127.5;
    const offset = -1.0;
    const input = randomNormal([2, 4, 5, 3]);
    const expectedOutputTensor = add(mul(input, scale), offset);
    const scalingLayer = new Rescaling({scale, offset});
    const layerOutputTensor = scalingLayer.apply(input) as Tensor;
    expectTensorsClose(layerOutputTensor, expectedOutputTensor);
  });

  it('Recasts dtype to float32', () => {
    const scale = 1.0 / 127.5;
    const offset = -1.0;
    const intTensor = randomNormal([2, 4, 5, 3], 7, 2, 'int32');
    const expectedOutputTensor = add(mul(intTensor, scale), offset);
    const scalingLayer = new Rescaling({scale, offset});
    const outputTensor = scalingLayer.apply(intTensor) as Tensor;
    expect(outputTensor.dtype).toBe('float32'); 
    expectTensorsClose(outputTensor, expectedOutputTensor); 
  });

  it('Config holds correct name', () => {
    const scale = 1.0 / 127.5;
    const offset = -1.0;
    const scalingLayer = new Rescaling({scale, offset, name: 'Rescaling'});
    const config = scalingLayer.getConfig();
    expect(config.name).toEqual('Rescaling'); 
  });

});
