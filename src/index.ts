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

import * as tf from '@tensorflow/tfjs';

import {nodeFileSystemRouter} from './io/file_system';
import * as nodeIo from './io/index';
import {NodeJSKernelBackend} from './nodejs_kernel_backend';
import * as nodeVersion from './version';

// tslint:disable-next-line:no-require-imports
import bindings = require('bindings');
import {TFJSBinding} from './tfjs_binding';

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

// tslint:disable-next-line:no-require-imports
const pjson = require('../package.json');

tf.registerBackend('tensorflow', () => {
  return new NodeJSKernelBackend(
      bindings('tfjs_binding.node') as TFJSBinding, pjson.name);
}, 3 /* priority */);

const success = tf.setBackend('tensorflow');
if (!success) {
  throw new Error(`Could not initialize TensorFlow backend.`);
}

// Register the model saving and loading handlers for the 'file://' URL scheme.
tf.io.registerLoadRouter(nodeFileSystemRouter);
tf.io.registerSaveRouter(nodeFileSystemRouter);

import {ProgbarLogger} from './callbacks';
// Register the ProgbarLogger for Model.fit() at verbosity level 1.
tf.registerCallbackConstructor(1, ProgbarLogger);

export * from './node';
