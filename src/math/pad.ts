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

import {ENV} from '../environment';
import * as util from '../util';
import {operation} from './decorators';
import {Array1D, Array2D} from './ndarray';

export class Ops {
  /**
   * Pads a Array1D.
   *
   * This operation will pad an array according to the `paddings` you specify.
   *
   * This operation currently only implements the `CONSTANT` mode from
   * Tensorflow's `pad` operation.
   *
   * @param x The array to pad.
   * @param paddings A tuple of ints [padLeft, padRight], how much to pad on the
   *     left and right side of the array.
   * @param constantValue The scalar pad value to use. Defaults to 0.
   */
  @operation
  static pad1D(x: Array1D, paddings: [number, number], constantValue = 0):
      Array1D {
    util.assert(
        paddings.length === 2,
        'Invalid number of paddings. Must be length of 2.');
    return ENV.engine.executeKernel(
        'Pad1D', {inputs: {x}, args: {paddings, constantValue}});
  }

  /**
   * Pads a Array2D.
   *
   * This operation will pad an array according to the `paddings` you specify.
   *
   * This operation currently only implements the `CONSTANT` mode from
   * Tensorflow's `pad` operation.
   *
   * @param x The array to pad.
   * @param paddings A pair of tuple ints
   *     [[padTop, padBottom], [padLeft, padRight]], how much to pad on the
   *     array.
   * @param constantValue The scalar pad value to use. Defaults to 0.
   */
  @operation
  static pad2D(
      x: Array2D, paddings: [[number, number], [number, number]],
      constantValue = 0): Array2D {
    util.assert(
        paddings.length === 2 && paddings[0].length === 2 &&
            paddings[1].length === 2,
        'Invalid number of paddings. Must be length of 2 each.');
    return ENV.engine.executeKernel(
        'Pad2D', {inputs: {x}, args: {paddings, constantValue}});
  }
}
