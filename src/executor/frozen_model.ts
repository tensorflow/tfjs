/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as tfc from '@tensorflow/tfjs-core';

import {NamedTensorMap, NamedTensorsMap, tensorflow} from '../data/index';
import {OperationMapper} from '../operations/index';

import {GraphExecutor} from './graph_executor';

export class FrozenModel {
  private executor: GraphExecutor;
  private version = 'n/a';
  private weightManifest: tfc.io.WeightsManifestConfig;
  private pathPrefix: string;
  // Returns the version information for the tensorflow model GraphDef.
  get modelVersion(): string {
    return this.version;
  }
  /**
   * @param modelUrl url for the model file generated by scripts/convert.py
   * script.
   * @param weightManifestUrl url for the weight file generated by
   * scripts/convert.py script.
   * @param requestOption options for Request, which allows to send credentials
   * and custom headers.
   */
  constructor(
      private modelUrl: string, private weightManifestUrl: string,
      private requestOption?: RequestInit) {
    this.getPathPrefix();
  }

  private getPathPrefix() {
    const isAbsolute = /^[a-z][a-z0-9+.-]*:/.test(this.weightManifestUrl);
    if (isAbsolute) {
      const url = new URL(this.weightManifestUrl);
      const segments = url.pathname.split('/');
      segments.splice(-1);
      url.pathname = segments.join('/');
      this.pathPrefix = url.toString();
    } else {
      const segments = this.weightManifestUrl.split('/');
      segments.splice(-1);
      this.pathPrefix = segments.join('/');
    }
  }

  /**
   * Loads the model topology file and build the in memory graph of the model.
   */
  private async loadRemoteProtoFile(): Promise<tensorflow.GraphDef> {
    try {
      const response = await fetch(this.modelUrl, this.requestOption);
      return tensorflow.GraphDef.decode(
          new Uint8Array(await response.arrayBuffer()));
    } catch (error) {
      throw new Error(`${this.modelUrl} not found. ${error}`);
    }
  }

  /**
   * Loads and parses the weight manifest JSON file from the url, weight loader
   * uses the manifest config to download the set of weight files.
   */
  private async loadWeightManifest(): Promise<void> {
    try {
      const manifest = await fetch(this.weightManifestUrl, this.requestOption);
      this.weightManifest = await manifest.clone().json();
    } catch (error) {
      throw new Error(`${this.weightManifestUrl} not found. ${error}`);
    }
  }
  /**
   * Loads the model and weight files, construct the in memory weight map and
   * compile the inference graph.
   */
  async load(): Promise<boolean> {
    const graphPromise = this.loadRemoteProtoFile();
    const manifestPromise = this.loadWeightManifest();

    const [graph, ] = await Promise.all([graphPromise, manifestPromise]);

    this.version = `${graph.versions.producer}.${graph.versions.minConsumer}`;
    const weightMap = await tfc.io.loadWeights(
        this.weightManifest, this.pathPrefix, undefined, this.requestOption);
    this.executor =
        new GraphExecutor(OperationMapper.Instance.transformGraph(graph));
    this.executor.weightMap = this.convertTensorMapToTensorsMap(weightMap);
    return true;
  }

  /**
   * Executes infrerence for the model for given input tensors.
   * @param inputs tensor map of the inputs for the model, keyed by the input
   * node names.
   * @param outputs output node name from the Tensorflow model, if no outputs
   * are specified, the default outputs of the model would be used. You can
   * inspect intermediate nodes of the model by adding them to the outputs
   * array.
   *
   * @returns A single tensor if provided with a single output or no outputs are
   * provided and there is only one default output, otherwise return a tensor
   * map.
   */
  execute(inputs: NamedTensorMap, outputs?: string|string[]): tfc.Tensor
      |NamedTensorMap {
    if (this.executor.isControlFlowModel) {
      throw new Error(
          'The model contains control flow ops, ' +
          'please use executeAsync method');
    }
    const result = this.executor.execute(
        this.convertTensorMapToTensorsMap(inputs), outputs);
    const keys = Object.keys(result);
    return (keys.length === 1) ? result[keys[0]] : result;
  }

  /**
   * Executes inference for the model for given input tensors in async fashion,
   * use this method when your model contains control flow ops.
   * @param inputs tensor map of the inputs for the model, keyed by the input
   * node names.
   * @param outputs output node name from the Tensorflow model, if no outputs
   * are specified, the default outputs of the model would be used. You can
   * inspect intermediate nodes of the model by adding them to the outputs
   * array.
   *
   * @returns A Promise of single tensor if provided with a single output or no
   * outputs are provided and there is only one default output, otherwise return
   * a tensor map.
   */
  async executeAsync(inputs: NamedTensorMap, outputs?: string|string[]):
      Promise<tfc.Tensor|NamedTensorMap> {
    if (!this.executor.isControlFlowModel) {
      throw new Error(
          'The model does not contain control flow ops, ' +
          'please use execute method for better performance.');
    }
    const result = await this.executor.executeAsync(
        this.convertTensorMapToTensorsMap(inputs), outputs);
    const keys = Object.keys(result);
    return (keys.length === 1) ? result[keys[0]] : result;
  }

  private convertTensorMapToTensorsMap(map: NamedTensorMap): NamedTensorsMap {
    return Object.keys(map).reduce((newMap: NamedTensorsMap, key) => {
      newMap[key] = [map[key]];
      return newMap;
    }, {});
  }
  /**
   * Releases the memory used by the weight tensors.
   */
  dispose() {
    this.executor.dispose();
  }
}

/**
 * @param modelUrl url for the model file generated by scripts/convert.py
 * script.
 * @param weightManifestUrl url for the weight file generated by
 * scripts/convert.py script.
 * @param requestOption options for Request, which allows to send credentials
 * and custom headers.
 */
export async function loadFrozenModel(
    modelUrl: string, weightsManifestUrl: string,
    requestOption?: RequestInit): Promise<FrozenModel> {
  const model = new FrozenModel(modelUrl, weightsManifestUrl, requestOption);
  await model.load();
  return model;
}
