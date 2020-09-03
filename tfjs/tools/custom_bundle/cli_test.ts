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

import * as fs from 'fs';
import * as path from 'path';

describe('CLI binary', () => {
  it('should be present and executable', () => {
    const packageJsonPath = path.resolve(
        __dirname,
        '../../package.json',
    );
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'));
    const binEntry = packageJson.bin;
    expect(binEntry['tfjs-custom-bundle']).toBeDefined();

    const toolPath = path.resolve(
        path.dirname(packageJsonPath), binEntry['tfjs-custom-bundle']);

    expect(() => {
      fs.accessSync(toolPath, fs.constants.X_OK);
    }).not.toThrow();
  });
});
