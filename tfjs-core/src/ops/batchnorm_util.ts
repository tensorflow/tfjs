/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {Tensor, Tensor5D} from '../tensor';
import {Rank} from '../types';
import {reshape} from './reshape';

export function xAs5D<R extends Rank>(x: Tensor<R>) {
  let x5D: Tensor5D;
  if (x.rank === 0 || x.rank === 1) {
    x5D = reshape(x, [1, 1, 1, 1, x.size]);
  } else if (x.rank === 2) {
    x5D = reshape(x, [1, 1, 1, x.shape[0], x.shape[1]]);
  } else if (x.rank === 3) {
    x5D = reshape(x, [1, 1, x.shape[0], x.shape[1], x.shape[2]]);
  } else if (x.rank === 4) {
    x5D = reshape(x, [1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]]);
  } else {
    x5D = x as Tensor5D;
  }

  return x5D;
}
