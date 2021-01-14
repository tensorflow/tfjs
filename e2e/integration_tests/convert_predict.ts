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

/**
 * This file is 2/2 of the test suites for CUJ: convert->predict.
 *
 * This file does below things:
 *  - Load graph models using Converter api.
 *  - Load inputs.
 *  - Make inference using each backends, and validate the results against TF
 *    results.
 */
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import * as tfconverter from '@tensorflow/tfjs-converter';
import * as tfc from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';
import * as tfl from '@tensorflow/tfjs-layers';

import {CONVERT_PREDICT_MODELS, KARMA_SERVER, REGRESSION} from './constants';
import {createInputTensors} from './test_util';

const DATA_URL = 'convert_predict_data';

describeWithFlags(`${REGRESSION} convert_predict`, ALL_ENVS, () => {
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

  for (const modelType in CONVERT_PREDICT_MODELS) {
    const models =
        (CONVERT_PREDICT_MODELS as {[key: string]: string[]})[modelType];
    models.forEach(model => {
      it(`${model}.`, async () => {
        let inputsNames: string[];
        let inputsData: tfc.TypedArray[];
        let inputsShapes: number[][];
        let inputsDtypes: tfc.DataType[];
        let tfOutputNames: string[];
        let tfOutputData: tfc.TypedArray[];
        let tfOutputShapes: number[][];
        let tfOutputDtypes: tfc.DataType[];

        const fetches = [
          fetch(`${KARMA_SERVER}/${DATA_URL}/${model}.xs-data.json`)
              .then(response => response.json()),
          fetch(`${KARMA_SERVER}/${DATA_URL}/${model}.xs-shapes.json`)
              .then(response => response.json()),
          fetch(`${KARMA_SERVER}/${DATA_URL}/${model}.ys-data.json`)
              .then(response => response.json()),
          fetch(`${KARMA_SERVER}/${DATA_URL}/${model}.ys-shapes.json`)
              .then(response => response.json())
        ];
        if (modelType === 'graph_model') {
          fetches.push(
              ...[fetch(`${KARMA_SERVER}/${DATA_URL}/${model}.xs-name.json`)
                      .then(response => response.json()),
                  fetch(`${KARMA_SERVER}/${DATA_URL}/${model}.xs-dtype.json`)
                      .then(response => response.json()),
                  fetch(`${KARMA_SERVER}/${DATA_URL}/${model}.ys-name.json`)
                      .then(response => response.json()),
                  fetch(`${KARMA_SERVER}/${DATA_URL}/${model}.ys-dtype.json`)
                      .then(response => response.json())]);
        }
        [inputsData, inputsShapes, tfOutputData, tfOutputShapes, inputsNames,
         inputsDtypes, tfOutputNames, tfOutputDtypes] =
            await Promise.all(fetches);

        if (modelType === 'graph_model') {
          const numTensors = tfc.memory().numTensors;

          const $model = await tfconverter.loadGraphModel(
              `${KARMA_SERVER}/${DATA_URL}/${model}/model.json`);

          const namedInputs = createInputTensors(
                                  inputsData, inputsShapes, inputsDtypes,
                                  inputsNames) as tfc.NamedTensorMap;

          const result = await $model.executeAsync(namedInputs, tfOutputNames);

          const ys =
              ($model.outputs.length === 1 ? [result] : result) as tfc.Tensor[];

          // Validate outputs with tf results.
          for (let i = 0; i < ys.length; i++) {
            const y = ys[i];
            expect(y.shape).toEqual(tfOutputShapes[i]);
            expect(y.dtype).toEqual(tfOutputDtypes[i]);
            tfc.test_util.expectArraysClose(await y.data(), tfOutputData[i]);
          }

          // Dispose all tensors;
          Object.keys(namedInputs).forEach(key => namedInputs[key].dispose());
          ys.forEach(tensor => tensor.dispose());
          $model.dispose();

          expect(tfc.memory().numTensors).toEqual(numTensors);
        }
        if (modelType === 'layers_model') {
          const $model = await tfl.loadLayersModel(
              `${KARMA_SERVER}/${DATA_URL}/${model}/model.json`);

          const xs =
              createInputTensors(inputsData, inputsShapes) as tfc.Tensor[];

          const result = $model.predict(xs);

          const ys =
              ($model.outputs.length === 1 ? [result] : result) as tfc.Tensor[];

          // Validate outputs with keras results.
          for (let i = 0; i < ys.length; i++) {
            const y = ys[i];
            expect(y.shape).toEqual(tfOutputShapes[i]);
            tfc.test_util.expectArraysClose(
                await y.data(), tfOutputData[i], 0.005);
          }

          // Dispose all tensors;
          xs.forEach(tensor => tensor.dispose());
          ys.forEach(tensor => tensor.dispose());
        }
      });
    });
  }
});
