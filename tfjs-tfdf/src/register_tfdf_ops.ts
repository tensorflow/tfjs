/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {GraphNode, registerOp} from '@tensorflow/tfjs-converter';
import {scalar, tensor1d, tensor2d} from '@tensorflow/tfjs-core';

import * as tfdfWebAPIClient from './tfdf_web_api_client';
import {TFDFWebModelRunner} from './types/tfdf_web_model_runner';

type Assets = string|Blob;
let assets: Assets;
let modelRunner: TFDFWebModelRunner;

export function setAssets(newAssets: Assets) {
  assets = newAssets;
}

registerOp('SimpleMLCreateModelResource', () => {
  return [scalar(0)];
});

registerOp('SimpleMLLoadModelFromPathWithHandle', async (node: GraphNode) => {
  const tfdfWeb = await tfdfWebAPIClient.tfdfWeb();
  const loadOptions = {createdTFDFSignature: true};

  modelRunner = typeof assets === 'string' ?
      await tfdfWeb.loadModelFromUrl(assets, loadOptions) :
      await tfdfWeb.loadModelFromZipBlob(assets, loadOptions);

  return [scalar(0)];
});

registerOp('SimpleMLInferenceOpWithHandle', async (node: GraphNode) => {
  const inputs = node.inputs.map(input => input.arraySync());
  const denseOutputDim = node.attrs['dense_output_dim'] as number;

  const features = {
    numericalFeatures: inputs[0] as number[][],
    booleanFeatures: inputs[1] as number[][],
    categoricalIntFeatures: inputs[2] as number[][],
    categoricalSetIntFeaturesValues: inputs[3] as number[],
    categoricalSetIntFeaturesRowSplitsDim1: inputs[4] as number[],
    categoricalSetIntFeaturesRowSplitsDim2: inputs[5] as number[],
    denseOutputDim
  };

  const outputs = modelRunner.predictTFDFSignature(features);

  const densePredictionsTensor = tensor2d(outputs.densePredictions);
  const denseColRepresentationTensor =
      tensor1d(outputs.denseColRepresentation, 'string');

  return [densePredictionsTensor, denseColRepresentationTensor];
});
