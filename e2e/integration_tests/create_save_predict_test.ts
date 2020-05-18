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
import * as tfl from '@tensorflow/tfjs-layers';

import {BACKENDS, createInputTensors, LAYERS, LOCAL_SERVER, MODELS, REGRESSION} from './utils';

describe(`${REGRESSION} ${LAYERS} create_save_predict`, () => {
  MODELS.forEach(model => {
    describe(`${model}.`, () => {
      let inputsData: tfc.TypedArray[];
      let inputsShapes: number[][];
      let kerasOutputData: tfc.TypedArray[];
      let kerasOutputShapes: number[][];

      beforeAll(async () => {
        [inputsData, inputsShapes, kerasOutputData, kerasOutputShapes] =
            await Promise.all([
              await fetch(`${LOCAL_SERVER}/${model}.xs-data.json`)
                  .then(response => response.json()),
              await fetch(`${LOCAL_SERVER}/${model}.xs-shapes.json`)
                  .then(response => response.json()),
              await fetch(`${LOCAL_SERVER}/${model}.ys-data.json`)
                  .then(response => response.json()),
              await fetch(`${LOCAL_SERVER}/${model}.ys-shapes.json`)
                  .then(response => response.json())
            ]);
      });

      BACKENDS.forEach(backend => {
        it(`${model} with ${backend}.`, async () => {
          const $model =
              await tfl.loadLayersModel(`${LOCAL_SERVER}/${model}/model.json`);

          const xs = createInputTensors(inputsData, inputsShapes);

          tfc.setBackend(backend);

          const result = $model.predict(xs);

          const ys =
              ($model.outputs.length === 1 ? [result] : result) as tfc.Tensor[];

          // Validate outputs with keras results.
          for (let i = 0; i < ys.length; i++) {
            const y = ys[i];
            expect(y.shape).toEqual(kerasOutputShapes[i]);
            tfc.test_util.expectArraysClose(await y.data(), kerasOutputData[i]);
          }

          // Dispose all tensors;
          xs.forEach(tensor => tensor.dispose());
          ys.forEach(tensor => tensor.dispose());
        });
      });
    });
  });
});
