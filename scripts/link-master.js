#!/usr/bin/env node
// Copyright 2020 Google LLC. All Rights Reserved.
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

// This script updates package.json to link following packages at master:
// tfjs-core, tfjs-layers, tfjs-converter, tfjs-data and tfjs.

const fs = require('fs');

let pkg = fs.readFileSync('package.json', 'utf8');

// Link core.
pkg = `${pkg}`.replace(
    new RegExp(`"@tensorflow/tfjs-core": ".+"`, 'g'),
    `"@tensorflow/tfjs-core": "link:../tfjs-core"`);

// Link layers.
pkg = `${pkg}`.replace(
    new RegExp(`"@tensorflow/tfjs-layers": ".+"`, 'g'),
    `"@tensorflow/tfjs-layers": "link:../tfjs-layers"`);

// Link converter.
pkg = `${pkg}`.replace(
    new RegExp(`"@tensorflow/tfjs-converter": ".+"`, 'g'),
    `"@tensorflow/tfjs-converter": "link:../tfjs-converter"`);

// Link data.
pkg = `${pkg}`.replace(
    new RegExp(`"@tensorflow/tfjs-data": ".+"`, 'g'),
    `"@tensorflow/tfjs-data": "link:../tfjs-data"`);

// Link union package.
pkg = `${pkg}`.replace(
    new RegExp(`"@tensorflow/tfjs": ".+"`, 'g'),
    `"@tensorflow/tfjs": "link:../tfjs"`);

fs.writeFileSync('package.json', pkg);
