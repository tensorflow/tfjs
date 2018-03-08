/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import * as tf from 'deeplearn';

const oneTwentySeven = tf.scalar(127);
const one = tf.scalar(1);

/**
 * A class that wraps webcam video elements to capture Tensor4Ds.
 */
export class Webcam {
  constructor(private webcamElement: HTMLVideoElement) {}

  capture(): tf.Tensor4D {
    return tf.tidy(() => {
      // TODO(nsthorat): Remove expandDims when model.predict allows
      // non-batched inputs.
      const img =
          tf.fromPixels(this.webcamElement).expandDims(0) as tf.Tensor4D;

      return img.asType('float32').div(oneTwentySeven).sub(one);
    });
  }

  async setup(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      // tslint:disable-next-line:no-any
      const navigatorAny = navigator as any;
      navigator.getUserMedia = navigator.getUserMedia ||
          navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
          navigatorAny.msGetUserMedia;

      if (navigator.getUserMedia) {
        navigator.getUserMedia(
            {video: true},
            (stream) => {
              this.webcamElement.src = window.URL.createObjectURL(stream);
              this.webcamElement.addEventListener('loadeddata', async () => {
                resolve();
              }, false);
            },
            (error) => {
              console.error(error);
              reject();
            });
      } else {
        reject();
      }
    });
  }
}
