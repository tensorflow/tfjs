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
 *
 * =============================================================================
 */

// tslint:disable:max-line-length
import * as tf from '@tensorflow/tfjs-core';
import {TensorContainer, TensorContainerArray, TensorContainerObject} from '@tensorflow/tfjs-core/dist/tensor_types';

import {Dataset} from '.';
import {LazyIterator} from './iterators/lazy_iterator';
// tslint:enable:max-line-length

// TODO(soergel): clean up the |string union type throughout when Tensor
// supports string.

// TODO(soergel): consider factoring out the Tensor dependency here, to allow
// making a Datasets package not dependent on DLJS.

/**
 * The value associated with a given key for a single element.
 *
 * Such a value may not have a batch dimension.  A value may be a scalar or an
 * n-dimensional array.
 */
export type ElementArray = number|number[]|tf.Tensor|string;

/**
 * The value associated with a given key for a batch of elements.
 *
 * Such a value must always have a batch dimension, even if it is of length 1.
 */
export type BatchArray = tf.Tensor|string[];

/**
 * A map from string keys (aka column names) to values for a single element.
 */
export type TabularRecord = {
  // TODO(soergel): eliminate the need for TabularRecord.
  // (It's still an issue for BatchDataset and Statistics.)
  [key: string]: ElementArray
};

/**
 * JSON-like type representing a nested structure of primitives or Tensors.
 */
export type DataElement = TensorContainer;

export type DataElementObject = TensorContainerObject;

export type DataElementArray = TensorContainerArray;

export type PrimitiveOrT<T> = void|string|number|boolean|T;

// Maybe this should be called 'NestedContainer'-- that's just a bit unwieldy.
export type Container<T> = ContainerObject<T>|ContainerArray<T>;

export type ContainerOrT<T> = Container<T>|T;

export interface ContainerObject<T> {
  [x: string]: ContainerOrT<T>;
}
export interface ContainerArray<T> extends Array<ContainerOrT<T>> {}

/**
 * A nested structure of Datasets, used as the input to zip().
 */
export type DatasetContainer = Container<Dataset<DataElement>>;

/**
 * A nested structure of LazyIterators, used as the input to zip().
 */
export type IteratorContainer = Container<LazyIterator<DataElement>>;

/**
 * A map from string keys (aka column names) to values for a batch of elements.
 */
export type DatasetBatch = {
  [key: string]: BatchArray
};
