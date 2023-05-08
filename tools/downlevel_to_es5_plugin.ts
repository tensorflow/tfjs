/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import * as ts from 'typescript';
import * as path from 'path';


// Transform that is enabled for es5 bundling. It transforms existing ES2015
// prodmode output to ESM5 so that the resulting bundles are using ES5 format.
// Inspired by Angular's ng_package ES5 transform:
// https://github.com/angular/angular/blob/a92a89b0eb127a59d7e071502b5850e57618ec2d/packages/bazel/src/ng_package/rollup.config.js#L150-L170
export const downlevelToEs5Plugin = {
  name: 'downlevel-to-es5',
  transform: (code: string, filePath: string) => {
    const compilerOptions = {
      target: ts.ScriptTarget.ES5,
      module: ts.ModuleKind.ES2015,
      allowJs: true,
      sourceMap: true,
      downlevelIteration: true,
      importHelpers: true,
      mapRoot: path.dirname(filePath),
    };
    const {outputText, sourceMapText}
          = ts.transpileModule(code, {compilerOptions});
    return {
      code: outputText,
      map: JSON.parse(sourceMapText),
    };
  },
};
