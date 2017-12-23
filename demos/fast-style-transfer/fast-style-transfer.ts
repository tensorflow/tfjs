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

import {Array3D, ENV} from 'deeplearn';
import {PolymerElement, PolymerHTMLElement} from '../polymer-spec';
import {TransformNet} from './net';

// tslint:disable-next-line:variable-name
export const StyleTransferDemoPolymer: new () => PolymerHTMLElement =
    PolymerElement({
      is: 'styletransfer-demo',
      properties: {
        contentNames: Array,
        selectedContentName: String,
        styleNames: Array,
        selectedStyleName: String
      }
    });

export enum ApplicationState {
  IDLE = 1,
  TRAINING = 2
}

const CONTENT_NAMES = ['stata', 'face', 'diana', 'Upload from file'];
const STYLE_MAPPINGS: {[varName: string]: string} = {
  'Udnie, Francis Picabia': 'udnie',
  'The Scream, Edvard Munch': 'scream',
  'La Muse, Pablo Picasso': 'la_muse',
  'Rain Princess, Leonid Afremov': 'rain_princess',
  'The Wave, Katsushika Hokusai': 'wave',
  'The Wreck of the Minotaur, J.M.W. Turner': 'wreck'
};
const STYLE_NAMES = Object.keys(STYLE_MAPPINGS);

export class StyleTransferDemo extends StyleTransferDemoPolymer {
  // Polymer properties
  contentNames: string[];
  selectedContentName: string;
  styleNames: string[];
  selectedStyleName: string;
  status: string;
  applicationState: ApplicationState;

  private transformNet: TransformNet;

  // DOM Elements
  private contentImgElement: HTMLImageElement;
  private styleImgElement: HTMLImageElement;
  // tslint:disable-next-line:no-any
  private sizeSlider: any;

  private canvas: HTMLCanvasElement;

  private startButton: HTMLButtonElement;

  // tslint:disable-next-line:no-any
  private camDialog: any;
  private stream: MediaStream;
  private webcamVideoElement: HTMLVideoElement;
  private takePicButton: HTMLButtonElement;
  private closeModal: HTMLButtonElement;

  private fileSelect: HTMLButtonElement;

  ready() {
    // Initialize deeplearn.js stuff
    this.canvas = this.querySelector('#imageCanvas') as HTMLCanvasElement;

    // Initialize polymer properties
    this.applicationState = ApplicationState.IDLE;
    this.status = '';

    // Retrieve DOM for images
    this.contentImgElement =
        this.querySelector('#contentImg') as HTMLImageElement;
    this.styleImgElement = this.querySelector('#styleImg') as HTMLImageElement;

    // Render DOM for images
    this.contentNames = CONTENT_NAMES;
    this.selectedContentName = 'stata';
    this.contentImgElement.src = 'images/stata.jpg';
    this.contentImgElement.height = 250;

    this.styleNames = STYLE_NAMES;
    this.selectedStyleName = 'Udnie, Francis Picabia';
    this.styleImgElement.src = 'images/udnie.jpg';
    this.styleImgElement.height = 250;

    this.transformNet =
        new TransformNet(STYLE_MAPPINGS[this.selectedStyleName]);

    this.initWebcamVariables();

    // tslint:disable-next-line:no-any
    this.sizeSlider = this.querySelector('#sizeSlider') as any;
    this.sizeSlider.addEventListener(
        'immediate-value-change',
        // tslint:disable-next-line:no-any
        (event: any) => {
          this.styleImgElement.height = this.sizeSlider.immediateValue;
          this.contentImgElement.height = this.sizeSlider.immediateValue;
        });
    // tslint:disable-next-line:no-any
    this.sizeSlider.addEventListener('change', (event: any) => {
      this.styleImgElement.height = this.sizeSlider.immediateValue;
      this.contentImgElement.height = this.sizeSlider.immediateValue;
    });

    this.fileSelect = this.querySelector('#fileSelect') as HTMLButtonElement;
    // tslint:disable-next-line:no-any
    this.fileSelect.addEventListener('change', (event: any) => {
      const f: File = event.target.files[0];
      const fileReader: FileReader = new FileReader();
      fileReader.onload = ((e) => {
        const target: FileReader = e.target as FileReader;
        this.contentImgElement.src = target.result;
      });
      fileReader.readAsDataURL(f);
      this.fileSelect.value = '';
    });

    // Add listener to drop downs
    const contentDropdown = this.querySelector('#content-dropdown');
    // tslint:disable-next-line:no-any
    contentDropdown.addEventListener('iron-activate', (event: any) => {
      const selected: string = event.detail.selected as string;
      if (selected === 'Use webcam') {
        this.openWebcamModal();
      } else if (selected === 'Upload from file') {
        this.fileSelect.click();
      } else {
        this.contentImgElement.src = 'images/' + selected + '.jpg';
      }
    });

    const styleDropdown = this.querySelector('#style-dropdown');
    // tslint:disable-next-line:no-any
    styleDropdown.addEventListener('iron-activate', (event: any) => {
      this.styleImgElement.src =
          'images/' + STYLE_MAPPINGS[event.detail.selected] + '.jpg';
    });

    // Add listener to start
    this.startButton = this.querySelector('#start') as HTMLButtonElement;
    this.startButton.addEventListener('click', () => {
      (this.querySelector('#load-error-message') as HTMLElement).style.display =
          'none';
      this.startButton.textContent =
          'Starting style transfer.. Downloading + running model';
      this.startButton.disabled = true;
      this.transformNet.setStyle(STYLE_MAPPINGS[this.selectedStyleName]);

      this.transformNet.load()
          .then(() => {
            this.startButton.textContent = 'Processing image';
            this.runInference();
            this.startButton.textContent = 'Start Style Transfer';
            this.startButton.disabled = false;
          })
          .catch((error) => {
            console.log(error);
            this.startButton.textContent = 'Start Style Transfer';
            this.startButton.disabled = false;
            const errMessage =
                this.querySelector('#load-error-message') as HTMLElement;
            errMessage.textContent = error;
            errMessage.style.display = 'block';
          });
    });
  }

  private initWebcamVariables() {
    this.camDialog = this.querySelector('#webcam-dialog');
    this.webcamVideoElement =
        this.querySelector('#webcamVideo') as HTMLVideoElement;
    this.takePicButton =
        this.querySelector('#takePicButton') as HTMLButtonElement;
    this.closeModal = this.querySelector('#closeModal') as HTMLButtonElement;

    // Check if webcam is even available
    // tslint:disable-next-line:no-any
    const navigatorAny = navigator as any;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      const contentNames = CONTENT_NAMES.slice();
      contentNames.unshift('Use webcam');
      this.contentNames = contentNames;
    }

    this.closeModal.addEventListener('click', () => {
      this.stream.getTracks()[0].stop();
    });

    this.takePicButton.addEventListener('click', () => {
      const hiddenCanvas: HTMLCanvasElement =
          this.querySelector('#hiddenCanvas') as HTMLCanvasElement;
      const hiddenContext: CanvasRenderingContext2D =
          hiddenCanvas.getContext('2d');
      hiddenCanvas.width = this.webcamVideoElement.width;
      hiddenCanvas.height = this.webcamVideoElement.height;
      hiddenContext.drawImage(
          this.webcamVideoElement, 0, 0, hiddenCanvas.width,
          hiddenCanvas.height);
      const imageDataURL = hiddenCanvas.toDataURL('image/jpg');
      this.contentImgElement.src = imageDataURL;
      this.stream.getTracks()[0].stop();
    });
  }

  private openWebcamModal() {
    this.camDialog.open();
    navigator.getUserMedia(
        {video: true},
        (stream) => {
          this.stream = stream;
          this.webcamVideoElement.src = window.URL.createObjectURL(stream);
          this.webcamVideoElement.play();
        },
        (err) => {
          console.error(err);
        });
  }

  async runInference() {
    await ENV.math.scope(async () => {
      const preprocessed = Array3D.fromPixels(this.contentImgElement);
      const inferenceResult = await this.transformNet.predict(preprocessed);
      this.setCanvasShape(inferenceResult.shape);
      renderToCanvas(inferenceResult, this.canvas);
    });
  }

  private setCanvasShape(shape: number[]) {
    this.canvas.width = shape[1];
    this.canvas.height = shape[0];
    if (shape[1] > shape[0]) {
      this.canvas.style.width = '500px';
      this.canvas.style.height = (shape[0] / shape[1] * 500).toString() + 'px';
    } else {
      this.canvas.style.height = '500px';
      this.canvas.style.width = (shape[1] / shape[0] * 500).toString() + 'px';
    }
  }
}

function renderToCanvas(a: Array3D, canvas: HTMLCanvasElement) {
  const [height, width, ] = a.shape;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = a.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const k = i * 3;
    imageData.data[j + 0] = Math.round(255 * data[k + 0]);
    imageData.data[j + 1] = Math.round(255 * data[k + 1]);
    imageData.data[j + 2] = Math.round(255 * data[k + 2]);
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

document.registerElement(StyleTransferDemo.prototype.is, StyleTransferDemo);
