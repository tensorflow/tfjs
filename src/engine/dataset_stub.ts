/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';
// NOTE: It is necessary to import `TensorContainer` from dist currently,
// because it is not exposed in the public API of tfjs-core.
import {TensorContainer} from '@tensorflow/tfjs-core/dist/tensor_types';

/**
 * Stub interfaces and classes for testing tf.Model.fitDataset().
 *
 * TODO(cais, soergel): Remove this in favor of actual interfaces and classes
 *   when ready.
 */

export abstract class LazyIterator<T> {
  abstract async next(): Promise<IteratorResult<T>>;
}

export abstract class Dataset<T extends TensorContainer> {
  abstract async iterator(): Promise<LazyIterator<T>>;
  size: number;
}

export type TensorMap = {
  [name: string]: tfc.Tensor
};

export type TensorOrTensorMap = tfc.Tensor|TensorMap;
