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

import * as tfc from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
import * as tfl from '@tensorflow/tfjs-layers';

import {KARMA_SERVER, LAYERS_MODELS, REGRESSION} from './constants';
import {createInputTensors} from './test_util';

/** Directory that stores the model. */
const DATA_URL = 'create_save_predict_data';

/**
 *  This file is 3/3 of the test suites for CUJ: create->save->predict.
 *
 *  This file test below things:
 *  - Load layers models using Layers api.
 *  - Load inputs.
 *  - Make inference using each backends, and validate the results against
 *    Keras results.
 */
describeWithFlags(`${REGRESSION} create_save_predict`, ALL_ENVS, () => {
  let originalTimeout: number;

  beforeAll(() => {
    // This test needs more time to finish the async fetch, adjusting
    // jasmine timeout for this test to avoid flakiness. See jasmine
    // documentation for detail:
    // https://jasmine.github.io/2.0/introduction.html#section-42
    originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });

  afterAll(() => jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout);

  LAYERS_MODELS.forEach(model => {
    it(`${model}.`, async () => {
      let inputsData: tfc.TypedArray[];
      let inputsShapes: number[][];
      let kerasOutputData: tfc.TypedArray[];
      let kerasOutputShapes: number[][];

      [inputsData, inputsShapes, kerasOutputData, kerasOutputShapes] =
          await Promise.all([
            fetch(`${KARMA_SERVER}/${DATA_URL}/${model}.xs-data.json`)
                .then(response => response.json()),
            fetch(`${KARMA_SERVER}/${DATA_URL}/${model}.xs-shapes.json`)
                .then(response => response.json()),
            fetch(`${KARMA_SERVER}/${DATA_URL}/${model}.ys-data.json`)
                .then(response => response.json()),
            fetch(`${KARMA_SERVER}/${DATA_URL}/${model}.ys-shapes.json`)
                .then(response => response.json())
          ]);

      const $model = await tfl.loadLayersModel(
          `${KARMA_SERVER}/${DATA_URL}/${model}/model.json`);

      const xs = createInputTensors(inputsData, inputsShapes) as tfc.Tensor[];
      const result = $model.predict(xs);

      const ys =
          ($model.outputs.length === 1 ? [result] : result) as tfc.Tensor[];

      // Validate outputs with keras results.
      for (let i = 0; i < ys.length; i++) {
        const y = ys[i];
        expect(y.shape).toEqual(kerasOutputShapes[i]);
        tfc.test_util.expectArraysClose(
            await y.data(), kerasOutputData[i], 0.005);
      }

      // Dispose all tensors;
      xs.forEach(tensor => tensor.dispose());
      ys.forEach(tensor => tensor.dispose());
    });
  });
});
