// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import * as path from 'path';
import * as shell from 'shelljs';

if (!process.argv[2]) {
  console.log('Must specify output path');
  process.exit(1);
}

const targetScriptPath = './node_modules/deeplearn-docs/src/make-api.ts';

const input = path.resolve('./src/index.ts');
const pkg = path.resolve('./package.json');
const src = path.resolve('./src/');
const repo = path.resolve('./');
const bundle = path.resolve('./dist/deeplearn.js');
const github = 'https://github.com/PAIR-code/deeplearnjs';

const out = path.resolve(process.argv[2]);

// tslint:disable-next-line:max-line-length
shell.exec(`ts-node ${targetScriptPath} --in ${input} --package ${pkg} --src ${
    src} --bundle ${bundle} --github ${github} --out ${out} --repo ${repo}`);
