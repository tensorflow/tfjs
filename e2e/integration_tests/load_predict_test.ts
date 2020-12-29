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
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
import * as tfl from '@tensorflow/tfjs-layers';

import {KARMA_SERVER, SMOKE} from './constants';

/**
 *  This file is the test suites for CUJ: load->predict.
 *
 *  This file test below things:
 *  - Load layers or graph model.
 *  - Make inference using each backends.
 */
describe(`${SMOKE} load_predict`, () => {
  describeWithFlags(`layers_model`, ALL_ENVS, () => {
    let model: tfl.LayersModel;
    let inputs: tfc.Tensor;

    const expected = [
      -0.003578941337764263, 0.0028922036290168762, -0.002957976423203945,
      0.00955402385443449
    ];

    beforeAll(async () => {
      model = await tfl.loadLayersModel(
          `${KARMA_SERVER}/load_predict_data/layers_model/model.json`);
    });

    beforeEach(() => {
      inputs = tfc.tensor([86, 11, 62, 40, 36, 75, 82, 94, 67, 75], [1, 10]);
    });

    afterEach(() => {
      inputs.dispose();
    });

    it(`predict`, async () => {
      const result = model.predict(inputs) as tfc.Tensor;
      tfc.test_util.expectArraysClose(await result.data(), expected);
    });
  });

  describeWithFlags(`graph_model`, ALL_ENVS, async () => {
    let a: tfc.Tensor;

    const expected = [
      0.7567615509033203, -0.18349379301071167, 0.7567615509033203,
      -0.18349379301071167
    ];

    beforeEach(() => {
      a = tfc.tensor2d([1, 1, 1, 1], [2, 2], 'float32');
    });

    afterEach(() => {
      a.dispose();
    });

    it(`predict for old model.`, async () => {
      const model = await tfconverter.loadGraphModel(
          `${KARMA_SERVER}/load_predict_data/graph_model/model.json`);
      const result = await model.executeAsync({'Placeholder': a}) as tfc.Tensor;
      tfc.test_util.expectArraysClose(await result.data(), expected);
    });

    it(`predict for new model.`, async () => {
      const model = await tfconverter.loadGraphModel(
          `${KARMA_SERVER}/load_predict_data/graph_model/model_new.json`);
      const result = await model.executeAsync({'Placeholder': a}) as tfc.Tensor;
      tfc.test_util.expectArraysClose(await result.data(), expected);
    });
  });
});
