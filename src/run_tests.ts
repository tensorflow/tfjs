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

import './index';
import * as jasmine_util from '@tensorflow/tfjs-core/dist/jasmine_util';

Error.stackTraceLimit = Infinity;

// tslint:disable-next-line:no-require-imports
const jasmineCtor = require('jasmine');

// tslint:disable-next-line:no-require-imports
import bindings = require('bindings');
import {TFJSBinding} from './tfjs_binding';
import {NodeJSKernelBackend} from './nodejs_kernel_backend';

jasmine_util.setTestEnvs([{
  name: 'test-tensorflow',
  factory: () =>
      new NodeJSKernelBackend(bindings('tfjs_binding.node') as TFJSBinding),
  features: {}
}]);

const IGNORE_LIST: string[] = [
  // See https://github.com/tensorflow/tfjs/issues/161
  'depthwiseConv2D',  // Requires space_to_batch() for dilation > 1.
  'separableConv2d',  // Requires space_to_batch() for dilation > 1.
];

const runner = new jasmineCtor();
runner.loadConfig({
  spec_files: [
    'src/**/*_test.ts', 'node_modules/@tensorflow/tfjs-core/dist/**/*_test.js'
  ],
  random: false
});

if (process.env.JASMINE_SEED) {
  runner.seed(process.env.JASMINE_SEED);
}

const env = jasmine.getEnv();

// Filter method that returns boolean, if a given test should return.
env.specFilter = spec => {
  // Return false (skip the test) if the test is in the ignore list.
  for (let i = 0; i < IGNORE_LIST.length; ++i) {
    if (spec.getFullName().startsWith(IGNORE_LIST[i])) {
      return false;
    }
  }
  // Otherwise run the test.
  return true;
};

runner.execute();
