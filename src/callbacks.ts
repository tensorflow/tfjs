/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original source: keras/callbacks.py */

import {BaseCallback} from './base_callbacks';
import {Container} from './engine/topology';
import {Model} from './engine/training';

export abstract class Callback extends BaseCallback {
  /** Instance of `keras.models.Model`. Reference of the model being trained. */
  model: Model = null;

  setModel(model: Container): void {
    if (!(model instanceof Model)) {
      throw new Error('model must be a Model, not some other Container');
    }
    this.model = model as Model;
  }
}
