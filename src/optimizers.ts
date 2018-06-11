/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Optimizers.
 */

// tslint:disable:max-line-length
import {Optimizer, train} from '@tensorflow/tfjs-core';

import * as K from './backend/tfjs_backend';
// tslint:enable:max-line-length

import {ValueError} from './errors';

// Add (de)serialize()

// Porting note: This diverges from the PyKeras implementation and may need to
// change based on (de)serialization requirements.
export function getOptimizer(identifier: string): Optimizer {
  const optimizerMap: {[optimizerName: string]: () => Optimizer} = {
    'Adagrad': () => train.adagrad(.01),
    'Adadelta': () => train.adadelta(1.0, 0.95, K.epsilon()),
    'Adam': () => train.adam(.001, .9, .999, K.epsilon()),
    'Adamax': () => train.adamax(0.002, .9, .999, K.epsilon(), 0.0),
    'RMSProp': () => train.rmsprop(.001, .9, null, K.epsilon()),
    'SGD': () => train.sgd(.01)
  };
  optimizerMap['adagrad'] = optimizerMap['Adagrad'];
  optimizerMap['adadelta'] = optimizerMap['Adadelta'];
  optimizerMap['adam'] = optimizerMap['Adam'];
  optimizerMap['adamax'] = optimizerMap['Adamax'];
  optimizerMap['rmsprop'] = optimizerMap['RMSProp'];
  optimizerMap['sgd'] = optimizerMap['SGD'];

  if (identifier in optimizerMap) {
    return optimizerMap[identifier]();
  }
  throw new ValueError(`Unknown Optimizer ${identifier}`);
}
