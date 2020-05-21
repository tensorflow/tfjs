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

    const expected = [
      -0.003578941337764263, 0.0028922036290168762, -0.002957976423203945,
      0.00955402385443449
    ];

    beforeAll(async () => {
      model = await tfl.loadLayersModel(getModelUrl('layers_model'));
    });

    beforeEach(() => {
      inputs = tfc.tensor([86, 11, 62, 40, 36, 75, 82, 94, 67, 75], [1, 10]);
    });

    afterEach(() => {
      inputs.dispose();
    });

    BACKENDS.forEach(backend => {
      it(`predict with ${backend}.`, async () => {
        await tfc.setBackend(backend);
        const result = model.predict(inputs) as tfc.Tensor;
        tfc.test_util.expectArraysClose(await result.data(), expected);
      });
    });
  });

  describe('graph_model', () => {
    let model: tfconverter.GraphModel;
    let a: tfc.Tensor;

    const expected = [
      0.7567615509033203, -0.18349379301071167, 0.7567615509033203,
      -0.18349379301071167
    ];

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
        const result = await model.executeAsync(a) as tfc.Tensor;
        tfc.test_util.expectArraysClose(await result.data(), expected);
      });
    });
  });
});
