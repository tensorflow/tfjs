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
import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as path from 'path';

import {GraphModeGoldenData, TensorDetail} from '../integration_tests/types';

const GRAPH_MODEL_GOLDEN_DATA_DIR =
    './integration_tests/graph_model_golden_data/';

type GraphModelInputs = tf.Tensor|tf.Tensor[]|tf.NamedTensorMap;

interface GraphModelConfig {
  readonly name: string;
  readonly url: string;
  readonly fromTFHub?: boolean;
  readonly inputs: GraphModelInputs;
}

const MODEL_CONFIGS: GraphModelConfig[] = [
  {
    name: 'MobileNetV3_small_075',
    url:
        'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_075_224/classification/5/default/1',
    fromTFHub: true,
    inputs: tf.randomNormal([1, 224, 224, 3]),
  },
];

async function getTensorDetail(tensor: tf.Tensor): Promise<TensorDetail> {
  const data = await tensor.data();
  return {
    data: Array.from(data),
    shape: tensor.shape,
    dtype: tensor.dtype,
  };
}

async function getTensorDetails(tensors: tf.Tensor|tf.Tensor[]|
                                tf.NamedTensorMap) {
  if (tensors instanceof tf.Tensor) {
    return await getTensorDetail(tensors);
  }

  if (Array.isArray(tensors)) {
    return await Promise.all(tensors.map(getTensorDetail));
  }

  const details: Record<string, TensorDetail> = {};
  for (const [name, tensor] of Object.entries(tensors)) {
    details[name] = await getTensorDetail(tensor);
  }
  return details;
}

function writeGraphModelGoldenData(data: GraphModeGoldenData) {
  const filename = `${data.name}.golden.json`;

  if (!fs.existsSync(GRAPH_MODEL_GOLDEN_DATA_DIR)) {
    fs.mkdirSync(GRAPH_MODEL_GOLDEN_DATA_DIR, {recursive: true});
  }
  fs.writeFileSync(
      path.join(GRAPH_MODEL_GOLDEN_DATA_DIR, filename), JSON.stringify(data));

  return filename;
}

(async function main() {
  if (MODEL_CONFIGS.length === 0) {
    return;
  }

  const models = await Promise.all(MODEL_CONFIGS.map(
      ({url, fromTFHub}) => tf.loadGraphModel(url, {fromTFHub})));


  const goldenModelDataNames: string[] = [];
  tf.env().set('KEEP_INTERMEDIATE_TENSORS', true);
  for (let i = 0; i < MODEL_CONFIGS.length; ++i) {
    const model = models[i];
    const {inputs} = MODEL_CONFIGS[i];

    const predictDetails = await getTensorDetails(model.predict(inputs));

    const intermediateTensors = model.getIntermediateTensors();
    const intermediateDetails: Record<string, TensorDetail[]> = {};

    for (const [name, tensors] of Object.entries(intermediateTensors)) {
      const details = await Promise.all(tensors.map(getTensorDetail));
      intermediateDetails[name] = details;
    }

    model.disposeIntermediateTensors();
    goldenModelDataNames.push(writeGraphModelGoldenData({
      ...MODEL_CONFIGS[i],
      inputs: await getTensorDetails(inputs),
      predictDetails,
      intermediateDetails,
    }));
  }

  fs.writeFileSync(
      path.join(GRAPH_MODEL_GOLDEN_DATA_DIR, 'golden_model_data.json'),
      JSON.stringify(goldenModelDataNames));
}());
