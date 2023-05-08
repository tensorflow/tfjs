/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import * as tf from './index';

/**
 * Returns an image used in various image related tests as a 4d tensor.
 *
 * The image is 8x8 and looks like this:
 * https://drive.google.com/file/d/1Y0AsFZ2w9HsWgJfm8f2uDOGY7A4IHjcK/view?usp=sharing
 *
 */
export function getTestImageAsTensor4d() {
  return tf.tensor4d(
      [
        156, 100, 111, 255, 171, 120, 117, 255, 183, 138, 109, 255, 200, 155,
        98,  255, 212, 157, 75,  255, 224, 156, 55,  255, 241, 153, 43,  255,
        230, 133, 18,  255, 168, 129, 130, 255, 186, 152, 140, 255, 202, 174,
        137, 255, 220, 190, 128, 255, 230, 193, 104, 255, 241, 188, 82,  255,
        250, 177, 64,  255, 233, 148, 31,  255, 179, 176, 159, 255, 199, 198,
        168, 255, 211, 216, 162, 255, 225, 228, 151, 255, 235, 227, 128, 255,
        243, 220, 106, 255, 247, 201, 81,  255, 222, 164, 41,  255, 163, 208,
        187, 255, 179, 226, 194, 255, 184, 234, 181, 255, 191, 239, 165, 255,
        201, 236, 142, 255, 213, 230, 126, 255, 218, 210, 103, 255, 191, 170,
        61,  255, 108, 214, 202, 255, 119, 226, 206, 255, 121, 231, 192, 255,
        130, 235, 179, 255, 141, 232, 162, 255, 155, 226, 146, 255, 162, 206,
        127, 255, 135, 166, 86,  255, 55,  207, 212, 255, 62,  217, 213, 255,
        64,  219, 201, 255, 76,  225, 193, 255, 87,  220, 175, 255, 94,  206,
        156, 255, 98,  181, 135, 255, 71,  143, 97,  255, 18,  200, 224, 255,
        19,  204, 222, 255, 19,  203, 211, 255, 30,  209, 205, 255, 35,  200,
        186, 255, 37,  177, 164, 255, 39,  150, 141, 255, 15,  115, 105, 255,
        0,   193, 228, 255, 0,   192, 221, 255, 0,   189, 209, 255, 0,   194,
        204, 255, 4,   182, 186, 255, 3,   158, 162, 255, 6,   133, 140, 255,
        0,   102, 113, 255
      ],
      [1, 8, 8, 4]);
}
