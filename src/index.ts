/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

// This file lists all exports of TensorFlow.js Layers

import * as dl from 'deeplearn';

import * as backend from './backend/deeplearnjs_backend';
import {LayerExports, ModelExports} from './exports';
// TODO(cais): Use dl optimizers (b/73762416).
import * as optimizers from './optimizers';

// tslint:disable:max-line-length
export {Callback, CallbackList, CustomCallback, CustomCallbackConfig, Logs} from './callbacks';
export {Model} from './engine/training';

export {ModelAndWeightsConfig, Sequential} from './models';
export {SymbolicTensor} from './types';
export {dl};  // TODO(cais): Remove this export (b/74099819).
export {backend, optimizers};
// tslint:enable:max-line-length

export const model = ModelExports.model;
export const sequential = ModelExports.sequential;
export const loadModel = ModelExports.loadModel;
export const input = ModelExports.input;
export const inputLayer = ModelExports.inputLayer;

export const layers = LayerExports;
