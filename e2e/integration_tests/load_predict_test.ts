/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import * as tfconverter from '@tensorflow/tfjs-converter';
import * as tfc from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';

import {BACKENDS, SMOKE} from './constants';

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';

const USE_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder/model.json';

/**
 *  This file is the test suites for CUJ: load->predict.
 *
 *  This file test below things:
 *  - Load layers or graph model.
 *  - Make inference using each backends.
 */
describe(`${SMOKE} load_predict`, () => {
  describe('layers_model', () => {
    let model: tfl.LayersModel;
    let inputs: tfc.Tensor;

    beforeAll(async () => {
      model = await tfl.loadLayersModel(MOBILENET_MODEL_PATH);
    });

    beforeEach(() => {
      inputs = tfc.zeros([1, 224, 224, 3]);
    });

    afterEach(() => {
      inputs.dispose();
    });

    BACKENDS.forEach(backend => {
      it(`predict with ${backend}.`, async () => {
        await tfc.setBackend(backend);
        model.predict(inputs);
      });
    });
  });

  describe('graph_model', () => {
    let model: tfconverter.GraphModel;
    let indices: tfc.Tensor;
    let values: tfc.Tensor;

    beforeAll(async () => {
      model = await tfconverter.loadGraphModel(USE_MODEL_PATH);
    });

    beforeEach(() => {
      indices = tfc.tensor2d(
          [
            0,  0, 0,  1, 0, 2, 0, 3, 0, 4, 1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1,
            5,  2, 0,  2, 1, 2, 2, 2, 3, 2, 4, 3, 0, 3, 1, 3, 2, 3, 3, 3, 4,
            4,  0, 4,  1, 4, 2, 4, 3, 4, 4, 4, 5, 4, 6, 4, 7, 4, 8, 4, 9, 4,
            10, 4, 11, 5, 0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 6, 5, 7
          ],
          [41, 2], 'int32');
      values = tfc.tensor1d(
          [
            16,   60,  69,   825, 6,    819, 2704, 2901, 903, 318, 6,
            728,  446, 31,   19,  54,   379, 18,   37,   735, 54,  829,
            5459, 11,  221,  8,   373,  7,   9,    969,  7,   468, 6,
            184,  621, 7582, 949, 1803, 18,  1977, 6
          ],
          'int32');
    });

    afterEach(() => {
      indices.dispose();
      values.dispose();
    });

    BACKENDS.forEach(backend => {
      it(`predict with ${backend}.`, async () => {
        await tfc.setBackend(backend);
        await model.executeAsync({indices, values});
      });
    });
  });
});
