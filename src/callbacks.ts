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
import {Container} from './engine/container';
import {LayersModel} from './engine/training';

export abstract class Callback extends BaseCallback {
  /** Instance of `keras.models.Model`. Reference of the model being trained. */
  model: LayersModel = null;

  setModel(model: Container): void {
    if (!(model instanceof LayersModel)) {
      throw new Error('model must be a LayersModel, not some other Container');
    }
    this.model = model as LayersModel;
  }
}
