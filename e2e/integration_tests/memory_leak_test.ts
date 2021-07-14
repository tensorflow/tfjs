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
import '@tensorflow/tfjs-backend-webgpu';

import * as tfconverter from '@tensorflow/tfjs-converter';
import * as tfc from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {SMOKE} from './constants';

const HOST = 'http://example.org';
const MODEL_URL = `${HOST}/model.json`;

const CUSTOM_OP_MODEL = {
  node: [
    {
      name: 'Input',
      op: 'Placeholder',
      attr: {
        dtype: {
          type: 1,  // DT_FLOAT
        },
        shape: {shape: {dim: [{size: 4}]}}
      }
    },
    {name: 'CustomOp', op: 'CustomOp', input: ['Input'], attr: {}}
  ],
  versions: {producer: 1.0, minConsumer: 3}
};

const weightsManifest: tfc.io.WeightsManifestEntry[] =
    [{'name': 'Const', 'dtype': 'float32', 'shape': [1]}];

const CUSTOM_HTTP_MODEL_LOADER = {
  load: async () => {
    const bias = tfc.tensor1d([0], 'float32');
    return {
      modelTopology: CUSTOM_OP_MODEL,
      weightSpecs: weightsManifest,
      weightData: bias.dataSync(),
      format: 'tfjs-graph-model',
      generatedBy: '1.15',
      convertedBy: '1.3.1'
    };
  }
};

describeWithFlags(
    `${SMOKE} A custom op that calls unmodularized kernels and modularized ` +
        `kernels`,
    ALL_ENVS, () => {
      it('should have no memory leak in a model run.', async () => {
        const model = new tfconverter.GraphModel(MODEL_URL);

        spyOn(tfc.io, 'getLoadHandlers').and.returnValue([
          CUSTOM_HTTP_MODEL_LOADER
        ]);

        // A custom op that calls unmodularized kernels and modularized kernels.
        tfconverter.registerOp('CustomOp', (nodeValue) => {
          const x = nodeValue.inputs[0];
          const softMax = tfc.softmax(x);
          const clone = tfc.clone(softMax);
          return [tfc.reshape(clone, [2, 2])];
        });

        await model.load();

        const before = tfc.memory().numTensors;

        const input = tfc.tensor1d([1, 2, 3, 4]);
        const output = model.predict(input) as tfc.Tensor;

        tfc.test_util.expectArraysClose(await output.data(), [
          0.032058604061603546, 0.08714432269334793, 0.23688283562660217,
          0.6439142823219299
        ]);

        input.dispose();
        output.dispose();

        const after = tfc.memory().numTensors;

        expect(after).toEqual(before);
      });
    });
