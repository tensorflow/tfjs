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

import {ApplicationConfig} from './firebase_types';

/** Smoke tests run in PR and nightly builds. */
export const SMOKE = '#SMOKE';
/** Regression tests run in nightly builds. */
export const REGRESSION = '#REGRESSION';

/** Testing tags. */
export const TAGS = [SMOKE, REGRESSION];

/** Testing backends. */
export const BACKENDS = ['cpu', 'webgl'];

/** Testing models for CUJ: create -> save -> predict. */
export const MODELS = [
  'mlp', 'cnn', 'depthwise_cnn', 'simple_rnn', 'gru', 'bidirectional_lstm',
  'time_distributed_lstm', 'one_dimensional', 'functional_merge'
];

/** Local server address for testing browser to access local files. */
export const LOCAL_SERVER = 'http://127.0.0.1:8080';

/** Default Firebase config. Used to construct config with apiKey substitute. */
export const FIREBASE_CONFIG: ApplicationConfig = {
  apiKey: '',
  authDomain: 'jstensorflow.firebaseapp.com',
  databaseURL: 'https://jstensorflow-integration.firebaseio.com/',
  projectId: 'jstensorflow',
  storageBucket: 'jstensorflow.appspot.com',
  messagingSenderId: '433613381222'
};

export const testEnv = {
  localTest: true
};
