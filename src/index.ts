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

import * as dl from '@tensorflow/tfjs-core';

// tslint:disable:max-line-length
import * as backend from './backend/deeplearnjs_backend';
import {ConstraintExports, InitializerExports, LayerExports, ModelExports, RegularizerExports} from './exports';

export {Callback, CallbackList, CustomCallback, CustomCallbackConfig, Logs} from './callbacks';
export {Model, ModelCompileConfig, ModelEvaluateConfig, ModelFitConfig, ModelPredictConfig} from './engine/training';
export {GRUCellLayerConfig, GRULayerConfig, LSTMCellLayerConfig, LSTMLayerConfig, RNN, RNNLayerConfig, SimpleRNNCellLayerConfig, SimpleRNNLayerConfig} from './layers/recurrent';
export {ModelAndWeightsConfig, Sequential, SequentialConfig} from './models';
export {SymbolicTensor} from './types';
export {version as version_layers} from './version';

export {dl};  // TODO(cais): Remove this export (b/74099819).
export {backend};

// tslint:enable:max-line-length

export const model = ModelExports.model;
export const sequential = ModelExports.sequential;
export const loadModel = ModelExports.loadModel;
export const input = ModelExports.input;
export const inputLayer = ModelExports.inputLayer;

export const layers = LayerExports;

export const constraints = ConstraintExports;
export const initializers = InitializerExports;
export const regularizers = RegularizerExports;
