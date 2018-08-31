/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import * as concat_util from './concat_util';

describe('concat_util.assertConcatShapesMatch rank=3D', () => {
  it('Non-3D tensor x1', () => {
    const assertFn = () => {
      concat_util.assertParamsConsistent([[1], [1, 2, 3]], 1);
    };

    expect(assertFn).toThrow();
  });

  it('Non-3D tensor x2', () => {
    const assertFn = () => {
      concat_util.assertParamsConsistent([[1, 2, 3], [2, 3]], 1);
    };

    expect(assertFn).toThrow();
  });

  it('axis out of bound', () => {
    const assertFn = () => {
      concat_util.assertParamsConsistent([[1, 2, 3], [1, 2, 3]], 4);
    };

    expect(assertFn).toThrow();
  });

  it('non-axis shape mismatch', () => {
    const assertFn = () => {
      concat_util.assertParamsConsistent([[2, 3, 3], [2, 2, 4]], 2);
    };

    expect(assertFn).toThrow();
  });

  it('shapes line up', () => {
    const assertFn = () => {
      concat_util.assertParamsConsistent([[2, 3, 3], [2, 3, 4]], 2);
    };

    expect(assertFn).not.toThrow();
  });

  it('3 shapes, all line up', () => {
    const assertFn = () => {
      concat_util.assertParamsConsistent([[2, 3, 3], [2, 3, 4], [2, 3, 8]], 2);
    };
    expect(assertFn).not.toThrow();
  });

  it('3 shapes, 3rd shape does not line up', () => {
    const assertFn = () => {
      concat_util.assertParamsConsistent([[2, 5, 3], [2, 1, 3], [2, 1, 5]], 1);
    };
    expect(assertFn).toThrow();
  });
});

describe('concat_util.computeConcatOutputShape', () => {
  it('compute output shape, axis=0', () => {
    expect(concat_util.computeOutShape([[2, 2, 3], [1, 2, 3]], 0)).toEqual([
      3, 2, 3
    ]);
  });
});
