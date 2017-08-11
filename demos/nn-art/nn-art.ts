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
import {PolymerElement, PolymerHTMLElement} from '../polymer-spec';
import {ActivationFunction, ColorMode, CPPN} from './cppn';

const CANVAS_UPSCALE_FACTOR = 3;
const MAT_WIDTH = 30;
// Standard deviations for gaussian weight initialization.
const WEIGHTS_STDEV = .6;

// tslint:disable-next-line:variable-name
const NNArtPolymer: new () => PolymerHTMLElement = PolymerElement({
  is: 'nn-art', properties: {}
});

class NNArt extends NNArtPolymer {
  private cppn: CPPN;

  private inferenceCanvas: HTMLCanvasElement;

  private z1Scale: number;
  private z2Scale: number;
  private numLayers: number;

  ready() {
    this.inferenceCanvas =
        this.querySelector('#inference') as HTMLCanvasElement;
    this.cppn = new CPPN(this.inferenceCanvas);

    this.inferenceCanvas.style.width =
        this.inferenceCanvas.width * CANVAS_UPSCALE_FACTOR + 'px';
    this.inferenceCanvas.style.height =
        this.inferenceCanvas.height * CANVAS_UPSCALE_FACTOR + 'px';

    const currentColorElement =
        this.querySelector('#colormode') as HTMLInputElement;
    this.querySelector('#color-selector')!.addEventListener(
        // tslint:disable-next-line:no-any
        'click', (event: any) => {
          const colorMode =
              (event.target as HTMLElement).getAttribute('data-val') as
              ColorMode;
          currentColorElement.value = colorMode;
          this.cppn.setColorMode(colorMode);
        });
    this.cppn.setColorMode('rgb');

    const currentActivationFnElement =
        this.querySelector('#activation-fn') as HTMLInputElement;
    this.querySelector('#activation-selector')!.addEventListener(
        // tslint:disable-next-line:no-any
        'click', (event: any) => {
          const activationFn =
              (event.target as HTMLElement).getAttribute('data-val') as
              ActivationFunction;
          currentActivationFnElement.value = activationFn;
          this.cppn.setActivationFunction(activationFn);
        });
    this.cppn.setActivationFunction('tanh');

    const layersSlider =
        this.querySelector('#layers-slider') as HTMLInputElement;
    const layersCountElement =
        this.querySelector('#layers-count') as HTMLDivElement;
    layersSlider!.addEventListener('input', (event) => {
      // tslint:disable-next-line:no-any
      this.numLayers = parseInt((event as any).target.value, 10);
      layersCountElement.innerText = '' + this.numLayers;
      this.cppn.setNumLayers(this.numLayers);
    });
    this.numLayers = parseInt(layersSlider.value, 10);
    layersCountElement.innerText = '' + this.numLayers;
    this.cppn.setNumLayers(this.numLayers);

    const z1Slider = this.querySelector('#z1-slider') as HTMLInputElement;
    z1Slider.addEventListener('input', (event) => {
      // tslint:disable-next-line:no-any
      this.z1Scale = parseInt((event as any).target.value, 10);
      this.cppn.setZ1Scale(this.z1Scale);
    });
    this.z1Scale = parseInt(z1Slider.value, 10);
    this.cppn.setZ1Scale(this.z1Scale);

    const z2Slider = this.querySelector('#z2-slider') as HTMLInputElement;
    z2Slider.addEventListener('input', (event) => {
      // tslint:disable-next-line:no-any
      this.z2Scale = parseInt((event as any).target.value, 10);
      this.cppn.setZ2Scale(this.z1Scale);
    });
    this.z2Scale = parseInt(z2Slider.value, 10);
    this.cppn.setZ2Scale(this.z2Scale);

    const randomizeButton = this.querySelector('#random') as HTMLButtonElement;
    randomizeButton.addEventListener('click', () => {
      this.cppn.generateWeights(MAT_WIDTH, WEIGHTS_STDEV);
    });

    this.cppn.generateWeights(MAT_WIDTH, WEIGHTS_STDEV);
    this.cppn.start();
  }
}

document.registerElement(NNArt.prototype.is, NNArt);
