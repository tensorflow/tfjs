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

import * as dl from 'deeplearn';
import {bindTensorFlowBackend} from '.';

// BeforeEach?
bindTensorFlowBackend();

describe('matMul', () => {
  it('should work', () => {
    const t1 = dl.tensor2d([[1, 2], [3, 4]]);
    const t2 = dl.tensor2d([[5, 6], [7, 8]]);
    const result = t1.matMul(t2);
    expect(result.dataSync()).toEqual(new Float32Array([19, 22, 43, 50]));
  });
});

describe('pad', () => {
  it('should work', () => {
    const t = dl.tensor2d([[1, 1], [1, 1]]);
    const result = dl.pad2d(t, [[1, 1], [1, 1]]);
    expect(result.dataSync()).toEqual(new Float32Array([
      0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0
    ]));
  });
});
