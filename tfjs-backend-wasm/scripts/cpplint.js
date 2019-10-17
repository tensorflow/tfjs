#!/usr/bin/env node
// Copyright 2019 Google LLC. All Rights Reserved.
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

const shell = require('shelljs');
const {exec} = require('../../scripts/test-util');
const fs = require('fs');

const result = shell.find('src/cc').filter(
    fileName => fileName.endsWith('.cc') || fileName.endsWith('.h'));

console.log(`C++ linting files:`);
console.log(result);

const cwd = process.cwd() + '/src/cc'
console.log(cwd);

const filenameArgument = result.join(' ');
exec(`python tools/cpplint.py --root ${cwd} ${filenameArgument}`);
