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

  capture(): tf.Tensor3D {
    return tf.tidy(() => {
      const img = this.cropImage(tf.fromPixels(this.webcamElement));
      return img.asType('float32').div(oneTwentySeven).sub(one);
    });
  }

  private cropImage(img: tf.Tensor3D) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  }

  private adjustVideoSize(width: number, height: number) {
    const aspectRatio = width / height;
    if (width >= height) {
      this.webcamElement.width = aspectRatio * this.webcamElement.height;
    } else if (width < height) {
      this.webcamElement.height = this.webcamElement.width / aspectRatio;
    }
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
            stream => {
              this.webcamElement.src = window.URL.createObjectURL(stream);
              this.webcamElement.addEventListener('loadeddata', async () => {
                this.adjustVideoSize(
                    this.webcamElement.videoWidth,
                    this.webcamElement.videoHeight);
                resolve();
              }, false);
            },
            (error) => {
              console.error(error);
              reject(null);
            });
      } else {
        reject();
      }
    });
  }
}
