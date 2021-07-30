/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import {version_core} from './index';

describe('version', () => {
  it('version is contained', () => {
    let expected: string;
    // Due to the difference between esbuild and bazel we need to try both
    // reltive and full path to load the package.json file.
    try {
      // For esbuild, the package.json need to be loaded with relative path.
      // tslint:disable-next-line:no-require-imports
      expected = require('../package.json').version;
      console.log(expected);
    } catch (e) {
      // In bazel nodejs test, the package.json is exported as js_library with
      // name tfjs-core. require will need the full path.
      // tslint:disable-next-line:no-require-imports
      expected = require('tfjs-core/package.json').version;
    }
    expect(version_core).toBe(expected);
  });
});
