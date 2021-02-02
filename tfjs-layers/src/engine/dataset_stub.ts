/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Stub interfaces and classes for testing tf.LayersModel.fitDataset().
 *
 * TODO(cais, soergel): Remove this in favor of actual interfaces and classes
 *   when ready.
 */

export abstract class LazyIterator<T> {
  abstract next(): Promise<IteratorResult<T>>;
}

export abstract class Dataset<T> {
  abstract iterator(): Promise<LazyIterator<T>>;
  size: number;
}
