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

import * as tf from '@tensorflow/tfjs-core';
import {deepMap, DeepMapResult, isIterable} from './deep_map';

export function deepClone<T>(container: T): T {
  return deepMap(container, cloneIfTensor);
}

// tslint:disable-next-line: no-any
function cloneIfTensor(item: any): DeepMapResult {
  if (item instanceof tf.Tensor) {
    return ({value: item.clone(), recurse: false});
  } else if (isIterable(item)) {
    return {value: null, recurse: true};
  } else {
    return {value: item, recurse: false};
  }
}
