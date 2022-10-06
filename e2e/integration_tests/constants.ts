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

/** Smoke tests run in PR and nightly builds. */
export const SMOKE = '#SMOKE';
/** Regression tests run in nightly builds. */
export const REGRESSION = '#REGRESSION';

/** Testing tags. */
export const TAGS = [SMOKE, REGRESSION];

/** Testing backends. */
export const BACKENDS = ['cpu', 'webgl'];

/** Testing models for CUJ: create -> save -> predict. */
export const LAYERS_MODELS = [
  'mlp', 'cnn', 'depthwise_cnn', 'simple_rnn', 'gru', 'bidirectional_lstm',
  'time_distributed_lstm', 'one_dimensional', 'functional_merge'
];

export const CONVERT_PREDICT_MODELS = {
  graph_model: [
    'saved_model_v1', 'saved_model_v2', 'saved_model_v2_with_control_flow',
    'saved_model_with_conv2d', 'saved_model_with_prelu',
    'saved_model_v2_complex64', 'saved_model_v2_with_control_flow_v2',
    'saved_model_v2_with_tensorlist_ops', 'saved_model_v1_with_hashtable'
  ],
  layers_model: ['mobilenet']
};

/** Karma server directory serving local files. */
export const KARMA_SERVER = './base/integration_tests';
