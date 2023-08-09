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

/**
 *  Base class for Task models.
 */

/* Original source: keras_nlp/models/task.py */
import { Tensor, serialization } from '@tensorflow/tfjs-core';

import { NotImplementedError } from '../../../errors';

import { PipelineModel, PipelineModelArgs } from '../utils';
import { Backbone } from './backbone';
import { Preprocessor } from './preprocessor';
import { ModelCompileArgs } from '../../../engine/training';
import { LossOrMetricFn } from '../../../types';
import { deserializeKerasObject, serializeKerasObject } from '../../../utils/generic_utils';

export class Task extends PipelineModel {
  protected _backbone: Backbone;
  protected _preprocessor: Preprocessor;

  constructor(args: PipelineModelArgs) {
    super(args);
  }

  private checkForLossMismatch(
    loss: string|string[]|{[outputName: string]: string}|LossOrMetricFn|
          LossOrMetricFn[]|{[outputName: string]: LossOrMetricFn}
  ) {
    throw new NotImplementedError();
  }

  override compile(args: ModelCompileArgs): void {
    this.checkForLossMismatch(args.loss);
    super.compile(args);
  }

  override preprocessSamples(x: Tensor, y?: Tensor, sampleWeight?: Tensor):
      Tensor | [Tensor, Tensor] | [Tensor, Tensor, Tensor] {
    return this.preprocessor.apply(x, {y, sampleWeight}) as Tensor;
  }

  /**
   * A `LayersModel` instance providing the backbone submodel.
   */
  get backbone(): Backbone {
    return this._backbone;
  }

  set backbone(value: Backbone) {
    this._backbone = value;
  }

  /**
   * A `LayersModel` instance used to preprocess inputs.
   */
  get preprocessor(): Preprocessor {
    return this._preprocessor;
  }

  set preprocessor(value: Preprocessor) {
    this.includePreprocessing = value != null;
    this._preprocessor = value;
  }

  override getConfig(): serialization.ConfigDict {
    // Don't chain to super here. The default `getConfig()` for functional
    // models is nested and cannot be passed to our Task constructors.
    return {
      'backbone': serializeKerasObject(this.backbone),
      'preprocessor': serializeKerasObject(this.preprocessor),
      'name': this.name,
    };
  }

  static override fromConfig<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>,
    config: serialization.ConfigDict
  ): T {
    // The default `fromConfig()` for functional models will return a
    // vanilla `LayersModel`. We override it to get a subclass instance back.
    if ('backbone' in config && !(config['backbone'] instanceof Backbone)) {
      config["backbone"] =
        deserializeKerasObject(config['backbone'] as serialization.ConfigDict);
    }
    if ('preprocessor' in config &&
        !(config['preprocessor'] instanceof Preprocessor)) {
      config['preprocessor'] = deserializeKerasObject(
        config['preprocessor'] as serialization.ConfigDict
      );
    }
    return new cls(config)
  }

  static backboneCls<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>
  ): serialization.SerializableConstructor<T> {
    return null;
  }

  static preprocessorCls<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>
  ): serialization.SerializableConstructor<T> {
    return null;
  }

  static presets<T extends serialization.Serializable>(
    cls: serialization.SerializableConstructor<T>
  ) {
    return {};
  }

  getLayers() {
    throw new NotImplementedError();
  }

  override summary(
    lineLength?: number,
    positions?: number[],
    printFn?: (message?: any, ...optionalParams: any[]) => void
  ): void {
    throw new NotImplementedError();
  }
}
