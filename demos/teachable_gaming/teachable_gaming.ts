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

// tslint:disable-next-line:max-line-length
import {TopKImageClassifier} from '../../models/topk_image_classifier/topk_image_classifier';
import {Array3D, gpgpu_util, GPGPUContext, NDArrayMathGPU} from '../deeplearn';
import {PolymerElement, PolymerHTMLElement} from '../polymer-spec';

// tslint:disable-next-line:no-any
declare const Dosbox: any;

// tslint:disable-next-line:variable-name
export const TeachableGamingDemoPolymer: new () => PolymerHTMLElement =
    PolymerElement({
      is: 'teachablegaming-demo',
      properties: {
        selectedGameIndex: {type: Number, observer: 'loadDosbox'},
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
  private selectedIndex: number;
  private predictedIndex: number;
  private selectedGameIndex = 0;

  private webcamVideoElement: HTMLVideoElement;
  private addNewKeyDialog: HTMLElement;
  private classifier: TopKImageClassifier;
  private keyEventData: Array<{code: number, key: string}>;
  private dosbox: {onload: (path: string, command: string) => void};
  private games:
      Array<{name: string, path: string, command: string, img: string}>;
  private static readonly knnKValue = 5;
  private static readonly maxControls = 15;

  ready() {
    this.webcamVideoElement =
        this.querySelector('#webcamVideo') as HTMLVideoElement;
    this.addNewKeyDialog = this.$.addkeydialog;
    this.addNewKeyDialog.addEventListener('keydown', (event: KeyboardEvent) => {
      console.log(event);
      const newKeyData = {code: event.keyCode, key: event.code};
      const newKeyEventData = this.keyEventData.slice();
      newKeyEventData.push(newKeyData);
      this.keyEventData = newKeyEventData;
      // tslint:disable-next-line:no-any
      (this.addNewKeyDialog as any).close();
    });

    this.keyEventData = [
      {code: -1, key: 'No action'},
      {code: 38, key: 'ArrowUp'},
      {code: 40, key: 'ArrowDown'},
      {code: 37, key: 'ArrowLeft'},
      {code: 39, key: 'ArrowRight'},
    ];
    this.games = [
      {
        name: 'Doom',
        path: 'https://js-dos.com/cdn/upload/DOOM-@evilution.zip',
        command: './DOOM/DOOM.EXE',
        img: 'https://js-dos.com/cdn/DOOM.png'
      },
      {
        name: 'Super Mario',
        path: 'https://js-dos.com/cdn/upload/mario-colin.zip',
        command: './Mario.exe',
        img: 'https://js-dos.com/cdn/mario.png'
      },
      {
        name: 'Donkey Kong',
        path: 'https://js-dos.com/cdn/upload/Donkey Kong 1983-@megalanya.zip',
        command: './dkong.exe',
        img: 'https://js-dos.com/cdn/Donkey%20Kong%201983.png'
      },
      {
        name: 'Tetris',
        path: 'https://js-dos.com/cdn/upload/Tetris-neozeed.zip',
        command: './',
        img: 'https://js-dos.com/cdn/Tetris.png'
      },
    ];
    this.selectedGameIndex = 0;

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
        TeachableGamingDemo.maxControls, TeachableGamingDemo.knnKValue,
        this.math);
    this.classifier.load();
    this.predictedIndex = -1;
    this.selectedIndex = -1;

    this.when(() => this.isDosboxReady(), () => this.loadDosbox());
    setTimeout(() => this.animate(), 1000);
  }

  private getKeyIndexFromId(id: string) {
    return parseInt(id.substring(id.indexOf('_') + 1), 10);
  }

  addNewKey() {
    // tslint:disable-next-line:no-any
    (this.addNewKeyDialog as any).open();
  }

  shouldDisableAddNewKey(keyEventData: Array<{}>) {
    return keyEventData.length > TeachableGamingDemo.maxControls;
  }

  toggle(event: Event) {
    const target = event.target as HTMLInputElement;
    const index = this.getKeyIndexFromId(target.id);

    const toggles = document.querySelectorAll('paper-toggle-button');
    if (target.checked) {
      this.selectedIndex = index;
      for (let i = 0; i < toggles.length; i++) {
        if (event.target !== toggles[i]) {
          (toggles[i] as HTMLInputElement).checked = false;
        }
      }
    } else {
      this.selectedIndex = -1;
    }
  }

  clear(event: Event) {
    const target = event.target as HTMLButtonElement;
    const index = this.getKeyIndexFromId(target.id);

    this.classifier.clearClass(index);
    const countBox = this.$$('#count_' + String(index));
    countBox.innerHTML = '0';
  }

  private isDosboxReady() {
    // tslint:disable-next-line:no-any
    return (window as any).Dosbox && (window as any).$;
  }

  private loadDosbox() {
    if (!this.isDosboxReady()) {
      return;
    }
    this.$.dosbox.innerHTML = '';
    this.dosbox = new Dosbox({
      id: 'dosbox',
      // tslint:disable-next-line:no-any
      onload: (dosbox: any) => {
        dosbox.run(
            this.games[this.selectedGameIndex].path,
            this.games[this.selectedGameIndex].command);
      },
      onrun: (dosbox: {}, app: string) => {
        console.log('App ' + app + ' is running');
      }
    });
    const newBackgroundPath =
        'url(' + this.games[this.selectedGameIndex].img + ')';
    (this.$$('.dosbox-overlay') as HTMLElement).style.background =
        newBackgroundPath;
  }

  private async animate() {
    if (this.selectedIndex >= 0) {
      await this.math.scope(async (keep, track) => {
        const image = track(Array3D.fromPixels(this.webcamVideoElement));
        const indicators = document.querySelectorAll('.indicators');
        for (let i = 0; i < indicators.length; i++) {
          (indicators[i] as HTMLElement).style.backgroundColor = 'lightgray';
        }
        await this.classifier.addImage(image, this.selectedIndex);
        const countBoxId = 'count_' + String(this.selectedIndex);
        const countBox = this.$$('#' + countBoxId) as HTMLElement;
        countBox.innerHTML = String(+countBox.innerHTML + 1);
      });
    } else if (this.$.predictswitch.checked) {
      await this.math.scope(async (keep, track) => {
        const image = track(Array3D.fromPixels(this.webcamVideoElement));
        const results = await this.classifier.predict(image);
        const indicators = document.querySelectorAll('.indicator');
        if (results.classIndex >= 0) {
          for (let i = 0; i < indicators.length; i++) {
            if (this.getKeyIndexFromId(indicators[i].id) ===
                results.classIndex) {
              (indicators[i] as HTMLElement).style.backgroundColor = 'green';
            } else {
              (indicators[i] as HTMLElement).style.backgroundColor =
                  'lightgray';
            }
          }
          const elem = this.$.dosbox;

          if (results.classIndex !== this.predictedIndex) {
            if (this.keyEventData[results.classIndex].code >= 0) {
              // tslint:disable-next-line:no-any
              const down = document.createEvent('Event') as any;
              down.initEvent('keydown', true, true);
              down.key = this.keyEventData[results.classIndex].key;
              down.keyCode = this.keyEventData[results.classIndex].code;
              elem.dispatchEvent(down);
            }

            if (this.predictedIndex !== -1 &&
                this.keyEventData[this.predictedIndex].code >= 0) {
              // tslint:disable-next-line: no-any
              const up = document.createEvent('Event') as any;
              up.initEvent('keyup', true, true);
              up.key = this.keyEventData[this.predictedIndex].key;
              up.keyCode = this.keyEventData[this.predictedIndex].code;
              elem.dispatchEvent(up);
            }
            this.predictedIndex = results.classIndex;
          }
        }
      });
    }

    setTimeout(() => this.animate(), 100);
  }

  getKeyIndicatorId(index: number) {
    return `indicator_${index}`;
  }

  getKeyToggleId(index: number) {
    return `toggle_${index}`;
  }

  getKeyClearId(index: number) {
    return `clear_${index}`;
  }

  getKeyCountId(index: number) {
    return `count_${index}`;
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
