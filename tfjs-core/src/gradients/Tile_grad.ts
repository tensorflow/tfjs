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

import {Tile, TileAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {add} from '../ops/add';
import {slice} from '../ops/slice';
import {zerosLike} from '../ops/zeros_like';
import {Tensor} from '../tensor';

export const tileGradConfig: GradConfig = {
  kernelName: Tile,
  inputsToSave: ['x'],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x] = saved;
    const {reps} = attrs as unknown as TileAttrs;

    const derX = () => {
      let xGrad = zerosLike(x);
      // TODO(cais): Maybe reduce memory footprint by avoiding repeated
      // slicing.
      if (x.rank === 1) {
        for (let i = 0; i < reps[0]; ++i) {
          xGrad = add(xGrad, slice(dy, [i * x.shape[0]], [x.shape[0]]));
        }
      } else if (x.rank === 2) {
        for (let i = 0; i < reps[0]; ++i) {
          for (let j = 0; j < reps[1]; ++j) {
            xGrad = add(xGrad, slice(dy, [i * x.shape[0], j * x.shape[1]], [
                          x.shape[0], x.shape[1]
                        ]));
          }
        }
      } else if (x.rank === 3) {
        for (let i = 0; i < reps[0]; ++i) {
          for (let j = 0; j < reps[1]; ++j) {
            for (let k = 0; k < reps[2]; ++k) {
              xGrad =
                  add(xGrad,
                      slice(
                          dy, [i * x.shape[0], j * x.shape[1], k * x.shape[2]],
                          [x.shape[0], x.shape[1], x.shape[2]]));
            }
          }
        }
      } else if (x.rank === 4) {
        for (let i = 0; i < reps[0]; ++i) {
          for (let j = 0; j < reps[1]; ++j) {
            for (let k = 0; k < reps[2]; ++k) {
              for (let l = 0; l < reps[3]; ++l) {
                xGrad =
                    add(xGrad,
                        slice(
                            dy,
                            [
                              i * x.shape[0], j * x.shape[1], k * x.shape[2],
                              l * x.shape[3]
                            ],
                            [x.shape[0], x.shape[1], x.shape[2], x.shape[3]]));
              }
            }
          }
        }
      } else {
        throw new Error(
            `Gradient for tile operation is not implemented for rank-` +
            `${x.rank} tensors yet.`);
      }
      return xGrad;
    };
    return {x: derX};
  },
};
