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

import { ModelPredictConfig, Scalar, Tensor, tensorScatterUpdate, tidy } from '@tensorflow/tfjs-core';

import { History } from '../../base_callbacks';
import { ContainerArgs } from '../../engine/container';
import { LayersModel, ModelEvaluateArgs } from '../../engine/training';
import { ModelFitArgs } from '../../engine/training_tensors';
import { NotImplementedError } from '../../errors';

export function tensorToArr(input: Tensor): unknown[] {
  return Array.from(input.dataSync()) as unknown as unknown[];
}

export function tensorArrTo2DArr(inputs: Tensor[]): unknown[][] {
  return inputs.map(input => tensorToArr(input));
}

/**
 * Returns a new Tensor with `updates` inserted into `inputs` starting at the
 * index `startIndices`.
 *
 * @param inputs Tensor to "modify"
 * @param startIndices the starting index to insert the slice.
 *  Length must be equal to `inputs.rank`;
 * @param updates the update tensor. Shape must fit within `inputs` shape.
 * @returns a new tensor with the modification.
 */
export function sliceUpdate(
    inputs: Tensor, startIndices: number[], updates: Tensor): Tensor {
  return tidy(() => {
    const indices: number[][] = [];
    /**
     * Computes the update indices by iterating through all indices from
     * `startIndices` to `startIndices + updates.shape`.
     */
    function createIndices(idx: number, curr: number[]): void {
      if (curr.length === startIndices.length) {
        indices.push(curr.slice());
        return;
      }
      const start = startIndices[idx];
      const end = start + updates.shape[idx];
      for (let i = start; i < end; i++) {
        curr.push(i);
        createIndices(idx + 1, curr);
        curr.pop();
      }
    }
    createIndices(0, []);
    // Flatten the updates to match length of its update indices.
    updates = updates.reshape([updates.size]);
    return tensorScatterUpdate(inputs, indices, updates);
  });
}

function packXYSampleWeight(x: Tensor, y?: Tensor, sampleWeight?: Tensor):
  Tensor
  | [Tensor, Tensor]
  | [Tensor, Tensor, Tensor] {
  throw new NotImplementedError();
}

function unPackXYSampleWeight(
  data: [Tensor]|[Tensor, Tensor]|[Tensor, Tensor, Tensor]
) {
  throw new NotImplementedError();
}

// TODO(pforderique): Figure out a workaround for `tf.data.Dataset`.
function convertInputsToDataset(
  x?: Tensor, y?: Tensor, sampleWeight?: Tensor, batchSize?: number
) {
  throw new NotImplementedError();
}

function trainValidationSplit(arrays: Tensor[], validationSplit: number) {
  throw new NotImplementedError();
}

/**
 * A model which allows automatically applying preprocessing.
 */
export interface PipelineModelArgs extends ContainerArgs {
  /**
   * Defaults to true.
   */
  includePreprocessing?: boolean;
}

export class PipelineModel extends LayersModel {
  /** @nocollapse */
  static override className = 'PipelineModel';

  protected includePreprocessing: boolean;

  constructor(args: PipelineModelArgs) {
    super(args);
    this.includePreprocessing = args.includePreprocessing ?? true;
  }

  /**
   * An overridable function which preprocesses features.
   */
  preprocessFeatures(x: Tensor) {
    return x;
  }

  /**
   * An overridable function which preprocesses labels.
   */
  preprocessLabels(y: Tensor) {
    return y;
  }

  /**
   * An overridable function which preprocesses entire samples.
   */
  preprocessSamples(x: Tensor, y?: Tensor, sampleWeight?: Tensor):
    Tensor
    | [Tensor, Tensor]
    | [Tensor, Tensor, Tensor] {
    throw new NotImplementedError();
  }

  // ---------------------------------------------------------------------------
  // Below are overrides to LayersModel methods to apply the functions above.
  // ---------------------------------------------------------------------------
  override fit(
    x: Tensor|Tensor[]|{[inputName: string]: Tensor},
    y: Tensor|Tensor[]|{[inputName: string]: Tensor},
    args: ModelFitArgs = {}
  ): Promise<History> {
    throw new NotImplementedError(
      `Uses ${convertInputsToDataset}, ${trainValidationSplit} ` +
      `${packXYSampleWeight}, and ${unPackXYSampleWeight}`);
  }

  override evaluate(
    x: Tensor|Tensor[],
    y: Tensor|Tensor[],
    args?: ModelEvaluateArgs
  ): Scalar | Scalar[] {
    throw new NotImplementedError();
  }

  override predict(
    x: Tensor | Tensor[],
    args?: ModelPredictConfig
  ): Tensor | Tensor[] {
    throw new NotImplementedError();
  }

  override trainOnBatch(
    x: Tensor|Tensor[]|{[inputName: string]: Tensor},
    y: Tensor|Tensor[]|{[inputName: string]: Tensor},
    sampleWeight?: Tensor
  ): Promise<number|number[]> {
    throw new NotImplementedError();
  }

  override predictOnBatch(x: Tensor|Tensor[]): Tensor|Tensor[] {
    throw new NotImplementedError();
  }
}
