/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {Array3D} from 'deeplearn';
import {PolymerElement, PolymerHTMLElement} from './polymer-spec';

// tslint:disable-next-line
export let NDArrayImageVisualizerPolymer: new () => PolymerHTMLElement =
    PolymerElement({is: 'ndarray-image-visualizer', properties: {}});

export class NDArrayImageVisualizer extends NDArrayImageVisualizerPolymer {
  private canvas: HTMLCanvasElement;
  private canvasContext: CanvasRenderingContext2D;
  private imageData: ImageData;

  ready() {
    this.canvas = this.querySelector('#canvas') as HTMLCanvasElement;
    this.canvas.width = 0;
    this.canvas.height = 0;
    this.canvasContext =
        this.canvas.getContext('2d') as CanvasRenderingContext2D;
    this.canvas.style.display = 'none';
  }

  setShape(shape: number[]) {
    this.canvas.width = shape[1];
    this.canvas.height = shape[0];
  }

  setSize(width: number, height: number) {
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
  }

  saveImageDataFromNDArray(ndarray: Array3D) {
    this.imageData = this.canvasContext.createImageData(
        this.canvas.width, this.canvas.height);
    if (ndarray.shape[2] === 1) {
      this.drawGrayscaleImageData(ndarray);
    } else if (ndarray.shape[2] === 3) {
      this.drawRGBImageData(ndarray);
    }
  }

  drawRGBImageData(ndarray: Array3D) {
    let pixelOffset = 0;
    for (let i = 0; i < ndarray.shape[0]; i++) {
      for (let j = 0; j < ndarray.shape[1]; j++) {
        this.imageData.data[pixelOffset++] = ndarray.get(i, j, 0);
        this.imageData.data[pixelOffset++] = ndarray.get(i, j, 1);
        this.imageData.data[pixelOffset++] = ndarray.get(i, j, 2);
        this.imageData.data[pixelOffset++] = 255;
      }
    }
  }

  drawGrayscaleImageData(ndarray: Array3D) {
    let pixelOffset = 0;
    for (let i = 0; i < ndarray.shape[0]; i++) {
      for (let j = 0; j < ndarray.shape[1]; j++) {
        const value = ndarray.get(i, j, 0);
        this.imageData.data[pixelOffset++] = value;
        this.imageData.data[pixelOffset++] = value;
        this.imageData.data[pixelOffset++] = value;
        this.imageData.data[pixelOffset++] = 255;
      }
    }
  }

  draw() {
    this.canvas.style.display = '';
    this.canvasContext.putImageData(this.imageData, 0, 0);
  }
}
document.registerElement(
    NDArrayImageVisualizer.prototype.is, NDArrayImageVisualizer);
