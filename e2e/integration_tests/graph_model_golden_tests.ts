/**
 * @license
 * Copyright 2023 Google LLC.
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

import {GOLDEN, KARMA_SERVER} from './constants';
import * as GOLDEN_MODEL_DATA_FILENAMES from './graph_model_golden_data/filenames.json';
import {GraphModeGoldenData, TensorDetail} from './types';

/** Directory that stores the model golden data. */
const DATA_URL = 'graph_model_golden_data';
const INTERMEDIATE_NODE_TESTS_NUM = 5;

describeWithFlags(`${GOLDEN} graph_model_golden`, ALL_ENVS, (env) => {
  let originalTimeout: number;

  beforeAll(async () => {
    // This test needs more time to finish the async fetch, adjusting
    // jasmine timeout for this test to avoid flakiness. See jasmine
    // documentation for detail:
    // https://jasmine.github.io/2.0/introduction.html#section-42
    originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
    await tfc.setBackend(env.backendName);
  });

  afterAll(() => jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout);

  for (const goldenFilename of GOLDEN_MODEL_DATA_FILENAMES) {
    describe(goldenFilename, () => {
      it('model.predict(...)', async () => {
        const [modelGolden, model] = await loadModelGolden(goldenFilename);
        const outputs = model.predict(createGoldenInputTensors(modelGolden));
        await expectTensorsToEqualGoldens(outputs, modelGolden.outputDetails);
        tfc.dispose(outputs);
      });

      it('model.execute(...) with default outputs', async () => {
        const [modelGolden, model] = await loadModelGolden(goldenFilename);
        const outputs = model.execute(createGoldenInputTensors(modelGolden));
        await expectTensorsToEqualGoldens(outputs, modelGolden.outputDetails);
        tfc.dispose(outputs);
      });

      for (let batchId = 1; batchId <= INTERMEDIATE_NODE_TESTS_NUM; ++batchId) {
        it(`model.execute(...) with intermediate node names #${batchId}`,
           async () => {
             const [modelGolden, model] = await loadModelGolden(goldenFilename);
             const intermediateNodeNames =
                 Object.keys(modelGolden.intermediateDetails);

             // Validates the intermediate node tensor values and output values.
             // Every `INTERMEDIATE_NODE_TESTS_NUM` nodes in
             // `intermediateDetails` are chosen to be validated.
             const targetNodeNames = [
               ...intermediateNodeNames.filter(
                   (unused, i) =>
                       (i % INTERMEDIATE_NODE_TESTS_NUM) + 1 === batchId),
               ...model.outputs.map(output => output.name),
             ];

             const goldens = targetNodeNames.map((name) => {
               const details = modelGolden.intermediateDetails[name];
               if (details == null) {
                 throw new Error(
                     `Golden file is missing tensor details for ` +
                     `${name}`);
               }
               return details;
             });

             const outputs = model.execute(
                                 createGoldenInputTensors(modelGolden),
                                 targetNodeNames) as tfc.Tensor[];

             expect(outputs.length).toEqual(goldens.length);
             await expectTensorsToEqualGoldens(outputs, goldens);
             tfc.dispose(outputs);
           });
      }
    });
  }
});

async function loadModelGolden(goldenFilename: string) {
  const modelGoldenPromise: Promise<GraphModeGoldenData> =
      fetch(`${KARMA_SERVER}/${DATA_URL}/${goldenFilename}`)
          .then(response => response.json());
  const modelPromise = modelGoldenPromise.then((modelGolden) => {
    return tfconverter.loadGraphModel(modelGolden.url, {
      fromTFHub: modelGolden.fromTFHub,
    });
  });

  return Promise.all([modelGoldenPromise, modelPromise]);
}

async function expectTensorToEqualGolden(
    tensor: tfc.Tensor, golden: TensorDetail) {
  expect(tensor).toEqual(jasmine.anything());
  expect(golden).toEqual(jasmine.anything());

  expect(isTensorDetail(golden));
  expect(tensor.isDisposed).toEqual(false);
  expect(tensor.dtype).toEqual(golden.dtype);
  expect(tensor.shape).toEqual(golden.shape);
  tfc.test_util.expectArraysClose(Array.from(await tensor.data()), golden.data);
}

async function expectTensorsToEqualGoldens(
    tensors: tfc.Tensor|tfc.Tensor[]|tfc.NamedTensorMap,
    goldens: TensorDetail|TensorDetail[]|Record<string, TensorDetail>) {
  expect(tensors).toEqual(jasmine.anything());
  expect(goldens).toEqual(jasmine.anything());
  if (tensors instanceof tfc.Tensor) {
    await expectTensorToEqualGolden(tensors, goldens as TensorDetail);
  } else if (Array.isArray(tensors)) {
    expect(Array.isArray(goldens)).toEqual(true);
    const details = goldens as TensorDetail[];
    expect(tensors.length).toEqual(details.length);
    for (let i = 0; i < tensors.length; ++i) {
      await expectTensorToEqualGolden(tensors[i], details[i]);
    }
  } else {
    const detailMap = goldens as Record<string, TensorDetail>;
    expect(new Set(Object.keys(detailMap)))
        .toEqual(new Set(Object.keys(tensors)));
    for (const [name, detail] of Object.entries(detailMap)) {
      await expectTensorToEqualGolden(tensors[name], detail);
    }
  }
}

function isTensorDetail(x: any): x is TensorDetail {
  return x != null && typeof x === 'object' && 'dtype' in x &&
      typeof x.dtype === 'string' && 'shape' in x && Array.isArray(x.shape) &&
      'data' in x && Array.isArray(x.data);
}

function createGoldenInputTensors({inputs}: GraphModeGoldenData) {
  function toTensor({data, dtype, shape}: TensorDetail) {
    let typedArray: tfc.TypedArray;
    switch (dtype) {
      case 'bool':
        typedArray = Uint8Array.from(data);
        break;
      case 'float32':
        typedArray = Float32Array.from(data);
        break;
      case 'int32':
        typedArray = Int32Array.from(data);
        break;
      default:
        throw new Error(`Unsupported input tensor type ${dtype}`);
    }
    return tfc.tensor(typedArray, shape, dtype);
  }

  if (Array.isArray(inputs)) {
    return inputs.map(toTensor);
  }
  if (isTensorDetail(inputs)) {
    return toTensor(inputs as TensorDetail);
  }
  return Object.entries(inputs).reduce(
      (map: tfc.NamedTensorMap, [name, detail]) => {
        map[name] = toTensor(detail);
        return map;
      },
      {});
}
