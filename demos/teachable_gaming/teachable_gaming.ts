/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '../demo-header';
import '../demo-footer';
import {Array3D, gpgpu_util, GPGPUContext, NDArrayMathGPU} from '../deeplearn';
// tslint:disable-next-line:max-line-length
import {TopKImageClassifier} from '../../models/topk_image_classifier/topk_image_classifier';
import {PolymerElement, PolymerHTMLElement} from '../polymer-spec';

// tslint:disable-next-line:no-any
declare const Dosbox: any;

// tslint:disable-next-line:variable-name
export const TeachableGamingDemoPolymer: new () => PolymerHTMLElement =
    PolymerElement({
      is: 'teachablegaming-demo',
      properties: {
      }
    });

/**
 * NOTE: To use the webcam without SSL, use the chrome flag:
 * --unsafely-treat-insecure-origin-as-secure=\
 *     http://localhost:5432
 */

export class TeachableGamingDemo extends TeachableGamingDemoPolymer {
  private math: NDArrayMathGPU;
  private gl: WebGLRenderingContext;
  private gpgpu: GPGPUContext;
  private selectedIndex = -1;

  private webcamVideoElement: HTMLVideoElement;
  private classifier: TopKImageClassifier;
  private toggles: HTMLInputElement[];
  private countBoxes: HTMLElement[];
  private clears: HTMLElement[];
  private indicators: HTMLElement[];
  private keyEventData: Array<{code: number, key: string}>;
  private dosbox: {};
  private static readonly knnKValue = 5;

  ready() {
    this.webcamVideoElement =
        this.querySelector('#webcamVideo') as HTMLVideoElement;
    this.toggles = [this.$.upswitch, this.$.downswitch, this.$.leftswitch,
      this.$.rightswitch, this.$.spaceswitch, this.$.sswitch];
     this.countBoxes = [this.$.upcount, this.$.downcount, this.$.leftcount,
      this.$.rightcount, this.$.spacecount, this.$.scount];
    this.clears = [this.$.upclear, this.$.downclear, this.$.leftclear,
      this.$.rightclear, this.$.spaceclear, this.$.sclear];
    this.indicators = [this.$.upindicator, this.$.downindicator,
      this.$.leftindicator, this.$.rightindicator, this.$.spaceindicator,
      this.$.sindicator];
    this.keyEventData = [
      {code: 38, key: 'ArrowUp'},
      {code: 40, key: 'ArrowDown'},
      {code: 37, key: 'ArrowLeft'},
      {code: 39, key: 'ArrowRight'},
      {code: 32, key: 'Space'},
      {code: 83, key: 'KeyS'},
    ];

    // tslint:disable-next-line:no-any
    const navigatorAny = navigator as any;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia(
          {video: true},
          (stream) => {
            this.webcamVideoElement.src = window.URL.createObjectURL(stream);
          },
          (error) => {
            console.warn(error);
          });
    }

    this.gl = gpgpu_util.createWebGLContext(this.inferenceCanvas);
    this.gpgpu = new GPGPUContext(this.gl);
    this.math = new NDArrayMathGPU(this.gpgpu);
    this.classifier = new TopKImageClassifier(
      this.keyEventData.length, TeachableGamingDemo.knnKValue, this.math);
    this.classifier.load();

    this.when(() => this.isDosboxReady(), () => this.loadDosbox());
    setTimeout(() => this.animate(), 1000);
  }

  toggle(event: Event) {
    const target = event.target as HTMLInputElement;
    let index = -1;
    for (let i = 0; i < this.toggles.length; i++) {
      if (this.toggles[i] === target) {
        index = i;
        break;
      }
    }
    if (index === -1) {
      console.warn('error bad toggle');
      return;
    }

    if (target.checked) {
      this.selectedIndex = index;
      for (let i = 0; i < this.toggles.length; i++) {
        if (i !== index) {
          this.toggles[i].checked = false;
        }
      }
    } else {
      this.selectedIndex = -1;
    }
  }

  clear(event: Event) {
    const target = event.target as HTMLButtonElement;
    let index = -1;
    for (let i = 0; i < this.clears.length; i++) {
      if (this.clears[i] === target) {
        index = i;
        break;
      }
    }
    if (index === -1) {
      console.warn('error bad button');
      return;
    }
    this.classifier.clearClass(index);
    this.countBoxes[index].innerHTML = '0';
  }

  private isDosboxReady() {
    // tslint:disable-next-line:no-any
    return (window as any).Dosbox && (window as any).$;
  }

  private loadDosbox() {
    this.dosbox = new Dosbox({
      id: 'dosbox',
      // tslint:disable-next-line:no-any
      onload: (dosbox: any) => {
        dosbox.run('https://js-dos.com/cdn/upload/DOOM-@evilution.zip',
          './DOOM/DOOM.EXE');
      },
      onrun: (dosbox: {}, app: string) => {
        console.log('App ' + app + ' is running');
      }
    });
  }

  private async animate() {
    if (this.selectedIndex >= 0) {

      await this.math.scope(async (keep, track) => {
        const image = track(Array3D.fromPixels(this.webcamVideoElement));
        for (let i = 0; i < this.indicators.length; i++) {
          this.indicators[i].style.backgroundColor = 'lightgray';
        }
        await this.classifier.addImage(image, this.selectedIndex);
        this.countBoxes[this.selectedIndex].innerHTML = String(
          +this.countBoxes[this.selectedIndex].innerHTML + 1);
      });
    }
    else if (this.$.predictswitch.checked) {
      await this.math.scope(async (keep, track) => {
        const image = track(Array3D.fromPixels(this.webcamVideoElement));
        const results = await this.classifier.predict(image);
        if (results.classIndex >= 0) {
          for (let i = 0; i < this.indicators.length; i++) {
            if (i === results.classIndex) {
              this.indicators[i].style.backgroundColor = 'green';
            } else {
              this.indicators[i].style.backgroundColor = 'lightgray';
            }
          }
          const elem = this.$.dosbox;

          // tslint:disable-next-line:no-any
          const event = document.createEvent('Event') as any;
          event.initEvent('keydown', true, true);
          event.key = this.keyEventData[results.classIndex].key;
          event.keyCode = this.keyEventData[results.classIndex].code;
          elem.dispatchEvent(event);
          // tslint:disable-next-line:no-any
          const event2 = document.createEvent('Event') as any;
          event2.initEvent('keyup', true, true);
          event.key = this.keyEventData[results.classIndex].key;
          event.keyCode = this.keyEventData[results.classIndex].code;
          elem.dispatchEvent(event2);
        }
      });
    }

    setTimeout(() => this.animate(), 100);
  }

  // tslint:disable-next-line:no-any
  when(check: () => any, exec: () => void) {
    let cancelled = false;
    const attempt = () => {
      if (cancelled) {
        return;
      }
      if (check()) {
        exec();
      } else {
        requestAnimationFrame(attempt);
      }
    };
    attempt();
    return {cancel: () => cancelled = true};
  }
}

document.registerElement(TeachableGamingDemo.prototype.is, TeachableGamingDemo);
