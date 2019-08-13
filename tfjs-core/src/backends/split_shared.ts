/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {Tensor} from '../tensor';

/** Shared implementation of the split kernel across WebGL and CPU. */
export function split<T extends Tensor>(
    x: T, sizeSplits: number[], axis: number): T[] {
  const begin = new Array(x.rank).fill(0);
  const size = x.shape.slice();
  return sizeSplits.map(s => {
    size[axis] = s;
    const slice = x.slice(begin, size);
    begin[axis] += s;
    return slice;
  });
}
