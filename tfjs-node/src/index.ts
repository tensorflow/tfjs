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
 * =============================================================================
 */

// Register all kernels.
import './register_all_kernels';

import * as tf from '@tensorflow/tfjs';
import * as path from 'path';

import {ProgbarLogger} from './callbacks';
import {nodeFileSystemRouter} from './io/file_system';
import * as nodeIo from './io/index';
import {NodeJSKernelBackend} from './nodejs_kernel_backend';
import {TFJSBinding} from './tfjs_binding';
import * as nodeVersion from './version';

// tslint:disable-next-line:no-require-imports
const binary = require('@mapbox/node-pre-gyp');
const bindingPath =
    binary.find(path.resolve(path.join(__dirname, '/../package.json')));

// Check if the node native addon module exists.
// tslint:disable-next-line:no-require-imports
const fs = require('fs');
if (!fs.existsSync(bindingPath)) {
  throw new Error(
      `The Node.js native addon module (tfjs_binding.node) can not ` +
      `be found at path: ` + String(bindingPath) + `. \nPlease run command ` +
      `'npm rebuild @tensorflow/tfjs-node` +
      (String(bindingPath).indexOf('tfjs-node-gpu') > 0 ? `-gpu` : ``) +
      ` --build-addon-from-source' to ` +
      `rebuild the native addon module. \nIf you have problem with building ` +
      `the addon module, please check ` +
      `https://github.com/tensorflow/tfjs/blob/master/tfjs-node/` +
      `WINDOWS_TROUBLESHOOTING.md or file an issue.`);
}
// tslint:disable-next-line:no-require-imports
const bindings = require(bindingPath);

// Merge version and io namespaces.
export const version = {
  ...tf.version,
  'tfjs-node': nodeVersion.version
};
export const io = {
  ...tf.io,
  ...nodeIo
};

// Export all union package symbols
export * from '@tensorflow/tfjs';
export * from './node';

// tslint:disable-next-line:no-require-imports
const pjson = require('../package.json');

// Side effects for default initialization of Node backend.
tf.registerBackend('tensorflow', () => {
  return new NodeJSKernelBackend(bindings as TFJSBinding, pjson.name);
}, 3 /* priority */);

const success = tf.setBackend('tensorflow');
if (!success) {
  throw new Error(`Could not initialize TensorFlow backend.`);
}

// Register the model saving and loading handlers for the 'file://' URL scheme.
tf.io.registerLoadRouter(nodeFileSystemRouter);
tf.io.registerSaveRouter(nodeFileSystemRouter);

// Register the ProgbarLogger for Model.fit() at verbosity level 1.
tf.registerCallbackConstructor(1, ProgbarLogger);
