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

import {Tensor} from '../../tensor';

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
export type ElementArray = number|number[]|Tensor|string;

/**
 * The value associated with a given key for a batch of elements.
 *
 * Such a value must always have a batch dimension, even if it is of length 1.
 */
export type BatchArray = Tensor|string[];

/**
 * A map from string keys (aka column names) to values for a single element.
 */
export type DatasetElement = {
  [key: string]: ElementArray
};

/**
 * A map from string keys (aka column names) to values for a batch of elements.
 */
export type DatasetBatch = {
  [key: string]: BatchArray
};
