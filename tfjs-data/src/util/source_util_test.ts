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

import {ENV} from '@tensorflow/tfjs-core';
import {isLocalPath} from './source_util';

const nonPathString = 'iamnotlocalpath';
const testData = ENV.get('IS_BROWSER') ? new Blob([nonPathString]) :
                                         Buffer.from(nonPathString);

describe('source_util', () => {
  it('returns true if it is local path', () => {
    const result = isLocalPath('file://testfile');
    expect(result).toBeTruthy();
  });

  it('returns false if it is not local path', () => {
    const result = isLocalPath(nonPathString);
    expect(result).toBeFalsy();
  });

  it('returns false if it is not string type', () => {
    const result = isLocalPath(testData);
    expect(result).toBeFalsy();
  });
});
