/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 *
 * =============================================================================
 */

import {browser, cast, env, expandDims, image, reshape, tensor1d, Tensor1D, tensor2d, Tensor2D, Tensor3D, Tensor4D, tidy, util} from '@tensorflow/tfjs-core';
import {WebcamConfig} from '../types';
import {LazyIterator} from './lazy_iterator';

/**
 * Provide a stream of image tensors from webcam video stream. Only works in
 * browser environment.
 */
export class WebcamIterator extends LazyIterator<Tensor3D> {
  private isClosed = true;
  private stream: MediaStream;
  private resize = false;
  private cropSize: [number, number];
  private cropBox: Tensor2D;
  private cropBoxInd: Tensor1D;

  private constructor(
      protected readonly webcamVideoElement: HTMLVideoElement,
      protected readonly webcamConfig: WebcamConfig) {
    super();
    if (this.needToResize()) {
      this.resize = true;
      this.cropSize =
          [this.webcamConfig.resizeHeight, this.webcamConfig.resizeWidth];
      this.cropBoxInd = tensor1d([0], 'int32');
      if (this.webcamConfig.centerCrop) {
        // Calculate the box based on resizing shape.
        const widthCroppingRatio =
            this.webcamConfig.resizeWidth * 1.0 / this.webcamVideoElement.width;
        const heightCroppingRatio = this.webcamConfig.resizeHeight * 1.0 /
            this.webcamVideoElement.height;
        const widthCropStart = (1 - widthCroppingRatio) / 2;
        const heightCropStart = (1 - heightCroppingRatio) / 2;
        const widthCropEnd = widthCropStart + widthCroppingRatio;
        const heightCropEnd = heightCroppingRatio + heightCropStart;
        this.cropBox = tensor2d(
            [heightCropStart, widthCropStart, heightCropEnd, widthCropEnd],
            [1, 4]);
      } else {
        this.cropBox = tensor2d([0, 0, 1, 1], [1, 4]);
      }
    }
  }

  summary() {
    return `webcam`;
  }

  // Construct a WebcamIterator and start it's video stream.
  static async create(
      webcamVideoElement?: HTMLVideoElement, webcamConfig: WebcamConfig = {}) {
    if (env().get('IS_NODE')) {
      throw new Error(
          'tf.data.webcam is only supported in browser environment.');
    }

    if (!webcamVideoElement) {
      // If webcam video element is not provided, create a hidden video element
      // with provided width and height.
      webcamVideoElement = document.createElement('video');
      if (!webcamConfig.resizeWidth || !webcamConfig.resizeHeight) {
        throw new Error(
            'Please provide webcam video element, or resizeWidth and ' +
            'resizeHeight to create a hidden video element.');
      }
      webcamVideoElement.width = webcamConfig.resizeWidth;
      webcamVideoElement.height = webcamConfig.resizeHeight;
    }
    const webcamIterator = new WebcamIterator(webcamVideoElement, webcamConfig);

    // Call async function to initialize the video stream.
    await webcamIterator.start();

    return webcamIterator;
  }

  // Async function to start video stream.
  async start(): Promise<void> {
    if (this.webcamConfig.facingMode) {
      util.assert(
          (this.webcamConfig.facingMode === 'user') ||
              (this.webcamConfig.facingMode === 'environment'),
          () =>
              `Invalid webcam facing mode: ${this.webcamConfig.facingMode}. ` +
              `Please provide 'user' or 'environment'`);
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: this.webcamConfig.deviceId,
          facingMode: this.webcamConfig.facingMode ?
              this.webcamConfig.facingMode :
              'user',
          width: this.webcamVideoElement.width,
          height: this.webcamVideoElement.height
        }
      });
    } catch (e) {
      // Modify the error message but leave the stack trace intact
      e.message = `Error thrown while initializing video stream: ${e.message}`;
      throw e;
    }

    if (!this.stream) {
      throw new Error('Could not obtain video from webcam.');
    }

    // Older browsers may not have srcObject
    try {
      this.webcamVideoElement.srcObject = this.stream;
    } catch (error) {
      console.log(error);
      this.webcamVideoElement.src = window.URL.createObjectURL(this.stream);
    }
    // Start the webcam video stream
    this.webcamVideoElement.play();

    this.isClosed = false;

    return new Promise<void>(resolve => {
      // Add event listener to make sure the webcam has been fully initialized.
      this.webcamVideoElement.onloadedmetadata = () => {
        resolve();
      };
    });
  }

  async next(): Promise<IteratorResult<Tensor3D>> {
    if (this.isClosed) {
      return {value: null, done: true};
    }

    let img;
    try {
      img = browser.fromPixels(this.webcamVideoElement);
    } catch (e) {
      throw new Error(
          `Error thrown converting video to pixels: ${JSON.stringify(e)}`);
    }
    if (this.resize) {
      try {
        return {value: this.cropAndResizeFrame(img), done: false};
      } catch (e) {
        throw new Error(`Error thrown cropping the video: ${e.message}`);
      } finally {
        img.dispose();
      }
    } else {
      return {value: img, done: false};
    }
  }

  private needToResize() {
    // If resizeWidth and resizeHeight are provided, and different from the
    // width and height of original HTMLVideoElement, then resizing and cropping
    // is required.
    if (this.webcamConfig.resizeWidth && this.webcamConfig.resizeHeight &&
        (this.webcamVideoElement.width !== this.webcamConfig.resizeWidth ||
         this.webcamVideoElement.height !== this.webcamConfig.resizeHeight)) {
      return true;
    }
    return false;
  }

  // Cropping and resizing each frame based on config
  cropAndResizeFrame(img: Tensor3D): Tensor3D {
    return tidy(() => {
      const expandedImage: Tensor4D = expandDims(cast(img, 'float32'), (0));
      let resizedImage;
      resizedImage = image.cropAndResize(
          expandedImage, this.cropBox, this.cropBoxInd, this.cropSize,
          'bilinear');
      // Extract image from batch cropping.
      const shape = resizedImage.shape;
      return reshape(resizedImage, shape.slice(1) as [number, number, number]);
    });
  }

  // Capture one frame from the video stream, and extract the value from
  // iterator.next() result.
  async capture(): Promise<Tensor3D> {
    return (await this.next()).value;
  }

  // Stop the video stream and pause webcam iterator.
  stop(): void {
    const tracks = this.stream.getTracks();

    tracks.forEach(track => track.stop());

    try {
      this.webcamVideoElement.srcObject = null;
    } catch (error) {
      console.log(error);
      this.webcamVideoElement.src = null;
    }
    this.isClosed = true;
  }

  // Override toArray() function to prevent collecting.
  toArray(): Promise<Tensor3D[]> {
    throw new Error('Can not convert infinite video stream to array.');
  }
}
