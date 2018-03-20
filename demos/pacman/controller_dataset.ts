/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs';

/**
 * A datset for webcam controls which allows the user to add example Tensors for
 * particular labels. This object will concat them into two large xs and ys.
 */
export class ControllerDataset {
  xs: tf.Tensor4D;
  ys: tf.Tensor2D;

  constructor(private numClasses: number) {}

  addExample(example: tf.Tensor4D, label: number) {
    const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]), this.numClasses));

    if (this.xs == null) {
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));
      oldX.dispose();

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));
      oldY.dispose();

      y.dispose();
    }
  }
}
