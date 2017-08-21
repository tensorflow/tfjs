/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import '../demo-header';
import '../demo-footer';
// tslint:disable-next-line:max-line-length
import {gpgpu_util, GPGPUContext, NDArrayMathCPU, NDArrayMathGPU} from '../deeplearnjs';
import * as imagenet_util from '../models/imagenet_util';
import {SqueezeNet} from '../models/squeezenet';
import {PolymerElement, PolymerHTMLElement} from '../polymer-spec';

// tslint:disable-next-line:variable-name
export const ImagenetDemoPolymer: new () => PolymerHTMLElement =
    PolymerElement({
      is: 'imagenet-demo',
      properties: {
        layerNames: Array,
        selectedLayerName: String,
        inputNames: Array,
        selectedInputName: String
      }
    });

/**
 * NOTE: To use the webcam without SSL, use the chrome flag:
 * --unsafely-treat-insecure-origin-as-secure=\
 *     http://localhost:5432
 */

const IMAGE_SIZE = 227;
const TOP_K_CLASSES = 5;

const INPUT_NAMES = ['cat', 'dog1', 'dog2', 'beerbottle', 'piano', 'saxophone'];
export class ImagenetDemo extends ImagenetDemoPolymer {
  private math: NDArrayMathGPU;
  private mathCPU: NDArrayMathCPU;
  private gl: WebGLRenderingContext;
  private gpgpu: GPGPUContext;
  private renderGrayscaleChannelsCollageShader: WebGLShader;

  private squeezeNet: SqueezeNet;

  private webcamVideoElement: HTMLVideoElement;
  private staticImgElement: HTMLImageElement;

  private layerNames: string[];
  private selectedLayerName: string;
  private inputNames: string[];
  private selectedInputName: string;

  private inferenceCanvas: HTMLCanvasElement;

  ready() {
    this.inferenceCanvas =
        this.querySelector('#inference-canvas') as HTMLCanvasElement;
    this.staticImgElement =
        this.querySelector('#staticImg') as HTMLImageElement;
    this.webcamVideoElement =
        this.querySelector('#webcamVideo') as HTMLVideoElement;

    this.layerNames = [];
    this.selectedLayerName = 'conv_1';

    const inputDropdown = this.querySelector('#input-dropdown');
    // tslint:disable-next-line:no-any
    inputDropdown.addEventListener('iron-activate', (event: any) => {
      const selectedInputName = event.detail.selected;
      if (selectedInputName === 'webcam') {
        this.webcamVideoElement.style.display = '';
        this.staticImgElement.style.display = 'none';
      } else {
        this.webcamVideoElement.style.display = 'none';
        this.staticImgElement.style.display = '';
      }
      this.staticImgElement.src = 'images/' + event.detail.selected + '.jpg';
    });

    if (navigator.getUserMedia) {
      navigator.getUserMedia(
          {video: true},
          (stream) => {
            this.webcamVideoElement.src = window.URL.createObjectURL(stream);
            this.initWithWebcam();
          },
          (error) => {
            console.log(error);
            this.initWithoutWebcam();
          });
    } else {
      this.initWithoutWebcam();
    }

    this.gl = gpgpu_util.createWebGLContext(this.inferenceCanvas);
    this.gpgpu = new GPGPUContext(this.gl);
    this.math = new NDArrayMathGPU(this.gpgpu);
    this.mathCPU = new NDArrayMathCPU();

    this.squeezeNet = new SqueezeNet(this.gpgpu, this.math);
    this.squeezeNet.loadVariables().then(() => {
      requestAnimationFrame(() => this.animate());
    });

    this.renderGrayscaleChannelsCollageShader =
        imagenet_util.getRenderGrayscaleChannelsCollageShader(this.gpgpu);
  }

  private initWithoutWebcam() {
    this.inputNames = INPUT_NAMES;
    this.selectedInputName = 'cat';
    this.staticImgElement.src = 'images/cat.jpg';
    this.webcamVideoElement.style.display = 'none';
    this.staticImgElement.style.display = '';

    if (location.protocol !== 'https:') {
      (this.querySelector('#ssl-message') as HTMLElement).style.display =
          'block';
    }

    (this.querySelector('#webcam-message') as HTMLElement).style.display =
        'block';
  }

  private initWithWebcam() {
    const inputNames = INPUT_NAMES.slice();
    inputNames.unshift('webcam');
    this.inputNames = inputNames;
    this.selectedInputName = 'webcam';
  }

  private animate() {
    const startTime = performance.now();

    const isWebcam = this.selectedInputName === 'webcam';

    const canvasTextureShape: [number, number] = [IMAGE_SIZE, IMAGE_SIZE];
    const canvasTexture =
        this.math.getTextureManager().acquireTexture(canvasTextureShape);

    const element = isWebcam ? this.webcamVideoElement : this.staticImgElement;
    this.gpgpu.uploadPixelDataToTexture(canvasTexture, element);

    this.math.scope((keep, track) => {
      const preprocessedInput =
          track(this.squeezeNet.preprocessColorTextureToArray3D(
              canvasTexture, canvasTextureShape));

      const inferenceResult = this.squeezeNet.infer(preprocessedInput);
      const namedActivations = inferenceResult.namedActivations;

      this.layerNames = Object.keys(namedActivations);
      this.layerNames.forEach(layerName => track(namedActivations[layerName]));

      const topClassesToProbability =
          this.squeezeNet.getTopKClasses(inferenceResult.logits, TOP_K_CLASSES);

      let count = 0;
      for (const className in topClassesToProbability) {
        if (!(className in topClassesToProbability)) {
          continue;
        }
        document.getElementById('class' + count).innerHTML = className;
        document.getElementById('prob' + count).innerHTML =
            '' + Math.floor(1000 * topClassesToProbability[className]) / 1000;
        count++;
      }

      const endTime = performance.now();

      (this.querySelector('#totalTime') as HTMLDivElement).innerHTML =
          'last inference time: ' +
          Math.floor(1000 * (endTime - startTime)) / 1000 + 'ms';

      // Render activations.
      const activationNDArray = namedActivations[this.selectedLayerName];

      // Compute max and min per channel for normalization.
      const maxValues = this.math.maxPool(
          activationNDArray, activationNDArray.shape[1],
          activationNDArray.shape[1], 0);
      const minValues = this.math.minPool(
          activationNDArray, activationNDArray.shape[1],
          activationNDArray.shape[1], 0);

      // Logically resize the rendering canvas. The displayed width is fixed.
      const imagesPerRow = Math.ceil(Math.sqrt(activationNDArray.shape[2]));
      const numRows = Math.ceil(activationNDArray.shape[2] / imagesPerRow);
      this.inferenceCanvas.width = imagesPerRow * activationNDArray.shape[0];
      this.inferenceCanvas.height = numRows * activationNDArray.shape[0];

      imagenet_util.renderGrayscaleChannelsCollage(
          this.gpgpu, this.renderGrayscaleChannelsCollageShader,
          activationNDArray.getTexture(), minValues.getTexture(),
          maxValues.getTexture(), activationNDArray.getTextureShapeRC(),
          activationNDArray.shape[0], activationNDArray.shape[2],
          this.inferenceCanvas.width, numRows);
    });

    this.math.getTextureManager().releaseTexture(
        canvasTexture, canvasTextureShape);

    requestAnimationFrame(() => this.animate());
  }
}

document.registerElement(ImagenetDemo.prototype.is, ImagenetDemo);
