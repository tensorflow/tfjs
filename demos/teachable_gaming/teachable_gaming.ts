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
import {Array3D, ENV, NDArrayMath} from 'deeplearn';
import {KNNImageClassifier} from 'deeplearn-knn-image-classifier';
import {PolymerElement, PolymerHTMLElement} from '../polymer-spec';

// tslint:disable-next-line:no-any
declare const Dosbox: any;

/**
 * Circular buffer to track a set of numbers and return the average over
 * the last n of those numbers. Used for performance calculations.
 */
class CircularBuffer {
  private arr: number[];
  private currentIndex = 0;
  private numEntries = 0;

  constructor(private n: number) {
    this.arr = new Array(this.n);
  }

  add(num: number) {
    this.arr[this.currentIndex] = num;
    this.currentIndex = (this.currentIndex + 1) % this.n;
    this.numEntries = Math.max(this.numEntries, this.currentIndex + 1);
  }

  getAverage(): number {
    const total = this.arr.reduce((sum: number, val: number) => {
      if (val == null) {
        return sum;
      }
      return sum + val;
    }, 0);
    return total / this.numEntries;
  }
}

// tslint:disable-next-line:variable-name
export const TeachableGamingDemoPolymer: new () => PolymerHTMLElement =
    PolymerElement({
      is: 'teachablegaming-demo',
      properties: {
        selectedGameIndex: {type: Number, observer: 'loadDosbox'},
        predicting: {type: Boolean}
      }
    });

/**
 * NOTE: To use the webcam without SSL, use the chrome flag:
 * --unsafely-treat-insecure-origin-as-secure=\
 *     http://localhost:5432
 */

export class TeachableGamingDemo extends TeachableGamingDemoPolymer {
  // Polymer properties.
  dosbox: {onload: (path: string, command: string) => void};
  predicting: boolean;
  selectedGameIndex = 0;

  private math: NDArrayMath;
  private selectedIndex: number;
  private predictedIndex: number;
  private hasAnyTrainedClass: boolean;
  private webcamVideoElement: HTMLVideoElement;
  private addNewKeyDialog: HTMLElement;
  private classifier: KNNImageClassifier;
  private keyEventData: Array<{code: number, key: string, text?: string}>;
  private games: Array<{
    name: string,
    path: string,
    command: string,
    img: string,
    keys: Array<{code: number, key: string, text?: string}>
  }>;
  private static readonly knnKValue = 5;
  private static readonly maxControls = 15;

  // Data members for tracking and displaying performance stats.
  private static readonly circularBufferSize = 20;
  private predictTimes: CircularBuffer;
  private animateLoopIndex: number;
  private static readonly animateLoopStatsFreq = 20;
  private previousFrameTime: number;
  private predictFps: CircularBuffer;
  private loggedEnv: boolean;

  ready() {
    this.math = ENV.math;
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

    this.keyEventData = [];
    this.games = [
      {
        name: 'Doom',
        path: 'https://js-dos.com/cdn/upload/DOOM-@evilution.zip',
        command: './DOOM/DOOM.EXE',
        img: 'https://js-dos.com/cdn/DOOM.png',
        keys: [
          {code: -1, key: 'No action'},
          {code: 38, key: 'ArrowUp', text: 'Forward'},
          {code: 40, key: 'ArrowDown', text: 'Back'},
          {code: 37, key: 'ArrowLeft', text: 'Left'},
          {code: 39, key: 'ArrowRight', text: 'Right'},
          {code: 87, key: 'KeyW', text: 'Use'},
          {code: 83, key: 'KeyS', text: 'Fire'},
          {code: 65, key: 'KeyA', text: 'Strafe left'},
          {code: 68, key: 'KeyD', text: 'Strafe right'},
          {code: 13, key: 'Enter'},
        ],
      },
      {
        name: 'Super Mario',
        path: 'https://js-dos.com/cdn/upload/mario-colin.zip',
        command: './Mario.exe',
        img: 'https://js-dos.com/cdn/mario.png',
        keys: [
          {code: -1, key: 'No action'},
          {code: 37, key: 'ArrowLeft', text: 'Left'},
          {code: 39, key: 'ArrowRight', text: 'Right'},
          {code: 18, key: 'AltLeft', text: 'Jump'},
        ],
      },
      {
        name: 'Tetris',
        path: 'https://js-dos.com/cdn/upload/Tetris-neozeed.zip',
        command: './',
        img: 'https://js-dos.com/cdn/Tetris.png',
        keys: [
          {code: -1, key: 'No action'},
          {code: 55, key: 'Digit7', text: 'Left'},
          {code: 56, key: 'Digit8', text: 'Right'},
          {code: 57, key: 'Digit9', text: 'Rotate'},
          {code: 32, key: 'Space', text: 'Drop'},
          {code: 13, key: 'Enter'},
        ],
      },
      {
        name: 'Duke Nukem 3D',
        path: 'https://js-dos.com/cdn/upload/Duke Nukem 3d-@digitalwalt.zip',
        command: './DUKE3D/DUKE3D.EXE',
        img: 'https://js-dos.com/cdn/Duke%20Nukem%203d.png',
        keys: [
          {code: -1, key: 'No action'},
          {code: 38, key: 'ArrowUp', text: 'Forward'},
          {code: 40, key: 'ArrowDown', text: 'Back'},
          {code: 37, key: 'ArrowLeft', text: 'Left'},
          {code: 39, key: 'ArrowRight', text: 'Right'},
          {code: 17, key: 'ControlRight', text: 'Fire'},
          {code: 65, key: 'KeyA', text: 'Jump'},
          {code: 13, key: 'Enter'},
        ],
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
            try {
              this.webcamVideoElement.srcObject = stream;
            } catch (error) {
              this.webcamVideoElement.src = window.URL.createObjectURL(stream);
            }
          },
          (error) => {
            console.warn(error);
          });
    }
    this.classifier = new KNNImageClassifier(
        TeachableGamingDemo.maxControls, TeachableGamingDemo.knnKValue,
        this.math);
    this.classifier.load();
    this.predictedIndex = -1;
    this.selectedIndex = -1;
    this.hasAnyTrainedClass = false;

    // Setup performance tracking vars.
    this.animateLoopIndex = 0;
    this.predictTimes =
        new CircularBuffer(TeachableGamingDemo.circularBufferSize);
    this.predictFps =
        new CircularBuffer(TeachableGamingDemo.circularBufferSize);
    this.predicting = false;
    this.loggedEnv = false;

    this.when(() => this.isDosboxReady(), () => this.loadDosbox());
    setTimeout(() => this.animate(), 1000);
  }

  private getIndexFromId(id: string) {
    return parseInt(id.substring(id.indexOf('_') + 1), 10);
  }

  addNewKey() {
    // tslint:disable-next-line:no-any
    (this.addNewKeyDialog as any).open();
  }

  shouldDisableAddNewKey(keyEventData: Array<{}>) {
    return keyEventData.length > TeachableGamingDemo.maxControls;
  }

  removeFocusFromButtons() {
    this.$.dosbox.focus();
  }

  getGameRadioId(index: number) {
    return 'game_' + String(index);
  }

  shouldRadioInitToChecked(index: number) {
    return index === 0;
  }

  onGameRadioClick(event: Event) {
    this.selectedGameIndex =
        this.getIndexFromId((event.target as HTMLElement).id);
  }

  toggle(event: Event) {
    const target = event.target as HTMLInputElement;
    const index = this.getIndexFromId(target.id);

    const toggles = document.querySelectorAll('.keytoggle');
    if (target.checked) {
      this.hasAnyTrainedClass = true;
      this.selectedIndex = index;
      for (let i = 0; i < toggles.length; i++) {
        if (event.target !== toggles[i]) {
          (toggles[i] as HTMLInputElement).checked = false;
        }
      }
    } else {
      this.selectedIndex = -1;
    }
    this.removeFocusFromButtons();
  }

  clear(event: Event) {
    const target = event.target as HTMLButtonElement;
    const index = this.getIndexFromId(target.id);

    this.classifier.clearClass(index);
    const countBox = this.$$('#count_' + String(index));
    countBox.innerHTML = '0';
    this.removeFocusFromButtons();
    this.hasAnyTrainedClass =
        this.classifier.getClassExampleCount().some(count => count !== 0);
  }

  private isDosboxReady() {
    // tslint:disable-next-line:no-any
    return (window as any).Dosbox && (window as any).$;
  }

  private loadDosbox() {
    if (!this.isDosboxReady()) {
      return;
    }
    this.keyEventData = this.games[this.selectedGameIndex].keys.slice();
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
    const frameTimeStart = performance.now();
    if (this.previousFrameTime != null) {
      this.predictFps.add(frameTimeStart - this.previousFrameTime);
      if (this.animateLoopIndex % TeachableGamingDemo.animateLoopStatsFreq ===
          0) {
        const fps = 1000 / this.predictFps.getAverage();
        this.$$('#predfps').innerHTML = String(fps.toFixed(3));
      }
    }
    this.previousFrameTime = frameTimeStart;
    if (this.selectedIndex >= 0) {
      this.predicting = false;
      await this.math.scope(async () => {
        const image = Array3D.fromPixels(this.webcamVideoElement);
        const indicators = document.querySelectorAll('.indicators');
        for (let i = 0; i < indicators.length; i++) {
          (indicators[i] as HTMLElement).style.backgroundColor = 'lightgray';
        }
        this.classifier.addImage(image, this.selectedIndex);
        const countBoxId = 'count_' + String(this.selectedIndex);
        const countBox = this.$$('#' + countBoxId) as HTMLElement;
        countBox.innerHTML = String(+countBox.innerHTML + 1);
      });
    } else if (this.hasAnyTrainedClass) {
      this.predicting = true;
      await this.math.scope(async () => {
        const image = Array3D.fromPixels(this.webcamVideoElement);
        const timeStart = performance.now();
        const results = await this.classifier.predictClass(image);
        this.predictTimes.add(performance.now() - timeStart);
        if (this.animateLoopIndex % TeachableGamingDemo.animateLoopStatsFreq ===
            0) {
          this.$$('#predperf').innerHTML =
              String(this.predictTimes.getAverage().toFixed(3));
        }
        const indicators = document.querySelectorAll('.indicator');
        if (results.classIndex >= 0) {
          for (let i = 0; i < indicators.length; i++) {
            if (this.getIndexFromId(indicators[i].id) === results.classIndex) {
              (indicators[i] as HTMLElement).style.backgroundColor = 'green';
            } else {
              (indicators[i] as HTMLElement).style.backgroundColor =
                  'lightgray';
            }
          }
          const elem = this.$.dosbox;

          if (this.$.predictswitch.checked) {
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
        }
      });
      // Log the environment first time through prediction.
      if (!this.loggedEnv) {
        console.log('Evaulated environment flags:');
        console.log(ENV);
        this.loggedEnv = true;
      }
    }
    this.animateLoopIndex++;
    requestAnimationFrame(() => this.animate());
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

  getKeyText(text: string) {
    if (!text) {
      return '-';
    }
    return '(' + text + ')';
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
