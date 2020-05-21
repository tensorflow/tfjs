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

import {BACKENDS, KARMA_SERVER, SMOKE} from './constants';

function getModelUrl(modelType: string) {
  return `${KARMA_SERVER}/load_predict_data/${modelType}/model.json`;
}

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
      model = await tfl.loadLayersModel(getModelUrl('layers_model'));
    });

    beforeEach(() => {
      inputs = tfc.zeros([1, 40, 40, 3]);
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
    let a: tfc.Tensor;

    beforeAll(async () => {
      model = await tfconverter.loadGraphModel(getModelUrl('graph_model'));
    });

    beforeEach(() => {
      a = tfc.tensor2d([1, 1, 1, 1], [2, 2], 'float32');
    });

    afterEach(() => {
      a.dispose();
    });

    BACKENDS.forEach(backend => {
      it(`predict with ${backend}.`, async () => {
        await tfc.setBackend(backend);
        await model.executeAsync(a);
      });
    });
  });
});
