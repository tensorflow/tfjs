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

import {ActivationFunction, ColorMode, CPPN} from '../nn-art/cppn';
import * as demo_util from '../util';

const inferenceCanvas =
    document.querySelector('#inference') as HTMLCanvasElement;

const isDeviceDisabled = demo_util.isSafari() && demo_util.isMobile();
const enableCPPN = demo_util.isWebGLSupported() && !isDeviceDisabled;

if (enableCPPN) {
  startCPPN();
} else {
  document.getElementById('disabled-demo-overlay').style.display = '';
  inferenceCanvas.style.display = 'none';
}

function startCPPN() {
  const DEFAULT_Z_SCALE = 1;
  const NUM_NEURONS = 30;
  const DEFAULT_NUM_LAYERS = 2;
  const WEIGHTS_STDEV = 0.6;

  const cppn = new CPPN(inferenceCanvas);

  cppn.setActivationFunction('tanh');
  cppn.setColorMode('rgb');
  cppn.setNumLayers(DEFAULT_NUM_LAYERS);
  cppn.setZ1Scale(convertZScale(DEFAULT_Z_SCALE));
  cppn.setZ2Scale(convertZScale(DEFAULT_Z_SCALE));
  cppn.generateWeights(NUM_NEURONS, WEIGHTS_STDEV);
  cppn.start();

  const currentColorElement =
      document.querySelector('#colormode') as HTMLInputElement;

  document.querySelector('#color-selector')
      .addEventListener(
          // tslint:disable-next-line:no-any
          'click', (event: any) => {
            const colorMode =
                (event.target as HTMLElement).getAttribute('data-val') as
                ColorMode;
            currentColorElement.value = colorMode;
            cppn.setColorMode(colorMode);
          });

  const currentActivationFnElement =
      document.querySelector('#activation-fn') as HTMLInputElement;
  document.querySelector('#activation-selector')
      .addEventListener(
          // tslint:disable-next-line:no-any
          'click', (event: any) => {
            const activationFn =
                (event.target as HTMLElement).getAttribute('data-val') as
                ActivationFunction;
            currentActivationFnElement.value = activationFn;
            cppn.setActivationFunction(activationFn);
          });

  const layersSlider =
      document.querySelector('#layers-slider') as HTMLInputElement;
  const layersCountElement =
      document.querySelector('#layers-count') as HTMLDivElement;
  layersSlider.addEventListener('input', (event) => {
    // tslint:disable-next-line:no-any
    const numLayers = parseInt((event as any).target.value, 10);
    layersCountElement.innerText = numLayers.toString();
    cppn.setNumLayers(numLayers);
  });
  layersCountElement.innerText = DEFAULT_NUM_LAYERS.toString();

  const z1Slider = document.querySelector('#z1-slider') as HTMLInputElement;
  z1Slider.addEventListener('input', (event) => {
    // tslint:disable-next-line:no-any
    const z1Scale = parseInt((event as any).target.value, 10);
    cppn.setZ1Scale(convertZScale(z1Scale));
  });

  const z2Slider = document.querySelector('#z2-slider') as HTMLInputElement;
  z2Slider.addEventListener('input', (event) => {
    // tslint:disable-next-line:no-any
    const z2Scale = parseInt((event as any).target.value, 10);
    cppn.setZ2Scale(convertZScale(z2Scale));
  });

  const randomizeButton =
      document.querySelector('#random') as HTMLButtonElement;
  randomizeButton.addEventListener('click', () => {
    cppn.generateWeights(NUM_NEURONS, WEIGHTS_STDEV);
    if (!playing) {
      cppn.start();
      requestAnimationFrame(() => {
        cppn.stopInferenceLoop();
      });
    }
  });

  let playing = true;
  const toggleButton = document.querySelector('#toggle') as HTMLButtonElement;
  toggleButton.addEventListener('click', () => {
    playing = !playing;
    if (playing) {
      toggleButton.innerHTML = 'STOP';
      cppn.start();
    } else {
      toggleButton.innerHTML = 'START';
      cppn.stopInferenceLoop();
    }
  });

  let canvasOnScreenLast = true;
  let scrollEventScheduled = false;
  const mainElement = document.querySelector('main') as HTMLElement;
  mainElement.addEventListener('scroll', () => {
    if (!scrollEventScheduled) {
      window.requestAnimationFrame(() => {
        const canvasOnScreen = isCanvasOnScreen();
        if (canvasOnScreen !== canvasOnScreenLast) {
          if (canvasOnScreen) {
            if (playing) {
              cppn.start();
            }
          } else {
            cppn.stopInferenceLoop();
          }
          canvasOnScreenLast = canvasOnScreen;
        }
        scrollEventScheduled = false;
      });
    }
    scrollEventScheduled = true;
  });

  function isCanvasOnScreen() {
    return mainElement.scrollTop < inferenceCanvas.offsetHeight;
  }

  function convertZScale(z: number): number {
    return (103 - z);
  }
}
