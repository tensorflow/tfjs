/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * The entry point for the webcam pacman demo, allowing a user to control a
 * JavaScript pacman game by training a model from the webcam which tie to
 * up / down / left / right controls.
 *
 * NOTE: This expects you to have run ./scripts/build-mobilenet-demo.sh
 */
import * as tf from 'deeplearn';
import * as tfl from 'tfjs-layers';

import {ControllerDataset} from './controller_dataset';
import {Webcam} from './webcam';

declare const PACMAN:
    {init(el: HTMLElement, path: string): void, keyDown(key: number): void;};
declare const KEY: {[key: string]: number};
declare const Pacman: {FPS: number};

// Fraction of examples collected to use for batch size.
const BATCH_SIZE_FRACTION = .4;
const LEARNING_RATE = .01;
const NUM_EPOCHS = 20;
const DENSE_UNITS = 100;

const NUM_CLASSES = 4;

const PACMAN_FPS = 10;
Pacman.FPS = PACMAN_FPS;

// This forces the buffer sub data async extension to turn off.
// It looks like in new versions of chrome this has been turned on,
// and deeplearn.js has a bug.
tf.ENV.set('WEBGL_GET_BUFFER_SUB_DATA_ASYNC_EXTENSION_ENABLED', false);

const CONTROLS = ['up', 'down', 'left', 'right'];
const CONTROL_CODES = ['ARROW_UP', 'ARROW_DOWN', 'ARROW_LEFT', 'ARROW_RIGHT'];

const webcamElement = document.getElementById('webcam') as HTMLVideoElement;
const webcam = new Webcam(webcamElement);

const costElement = document.getElementById('cost') as HTMLSpanElement;

let mobilenet: tfl.Model;
let model: tfl.Sequential;

const controllerDataset = new ControllerDataset(NUM_CLASSES);

async function train() {
  isPredicting = false;
  model = tfl.sequential({
    layers: [
      tfl.layers.flatten({inputShape: [7, 7, 256]}), tfl.layers.dense({
        units: getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'VarianceScaling',
        kernelRegularizer: 'L1L2',
        useBias: true,
        inputShape: [1000]
      }),
      tfl.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'VarianceScaling',
        kernelRegularizer: 'L1L2',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  const sgd = new tfl.optimizers.Adam({lr: getLearningRate()});
  model.compile({optimizer: sgd, loss: 'categoricalCrossentropy'});

  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  await model.fit({
    x: controllerDataset.xs,
    y: controllerDataset.ys,
    batchSize,
    epochs: getEpochs(),
    callbacks: {
      onBatchEnd: async (batch: number, logs: tfl.Logs) => {
        const cost = await (logs.loss as tf.Tensor).data();
        costElement.innerText = '' + cost[0];
      },
    },
  });
}

let isPredicting = false;
async function predict() {
  statusElement.style.visibility = 'visible';
  let lastTime = performance.now();
  while (isPredicting) {
    const prediction = tf.tidy(() => {
      const img = getActivation();
      return (model.predict(img) as tf.Tensor);
    });

    const classId = (await prediction.as1D().argMax().data())[0];
    const control = CONTROL_CODES[classId];
    fireEvent(control);

    const elapsed = performance.now() - lastTime;

    lastTime = performance.now();
    statusElement.innerText = CONTROLS[classId];
    document.getElementById('inferenceTime').innerText =
        'inference: ' + elapsed + 'ms';

    await tf.nextFrame();
  }
  statusElement.style.visibility = 'hidden';
}

function addExample(label: number) {
  tf.tidy(() => controllerDataset.addExample(getActivation(), label));
}

function getActivation(): tf.Tensor4D {
  return tf.tidy(() => mobilenet.predict(webcam.capture()) as tf.Tensor4D);
}

async function loadMobilenet(): Promise<tfl.Model> {
  console.log('Loading mobilenet...');
  // TODO(nsthorat): Move these to GCP when they are no longer JSON.
  const model = await tfl.loadModel('../../dist/demo/mobilenet/model.json');
  console.log('Done loading mobilenet.');

  // Return a model that outputs an internal activation.
  const layer = model.getLayer('conv_pw_13_relu');
  return tfl.model({inputs: model.inputs, outputs: layer.output});
}

let mouseDown = false;
const totals = [0, 0, 0, 0];
async function addExamples(label: number) {
  const className = CONTROLS[label];
  const button = document.getElementById(className);
  while (mouseDown) {
    addExample(label);
    button.innerText = className + ' (' + (totals[label]++) + ')';
    await tf.nextFrame();
  }
}
const upButton = document.getElementById('up');
const downButton = document.getElementById('down');
const leftButton = document.getElementById('left');
const rightButton = document.getElementById('right');

const thumbDisplayed: {[label: number]: boolean} = {};
const handler = (label: number) => {
  return () => {
    if (thumbDisplayed[label] == null) {
      const thumb = document.getElementById(CONTROLS[label] + '-thumb') as
          HTMLCanvasElement;
      const destCtx = thumb.getContext('2d');
      destCtx.drawImage(webcamElement, 0, 0, thumb.width, thumb.height);

      thumbDisplayed[label] = true;
    }

    mouseDown = true;
    addExamples(label);
  };
};

upButton.addEventListener('mousedown', handler(0));
upButton.addEventListener('mouseup', () => mouseDown = false);

downButton.addEventListener('mousedown', handler(1));
downButton.addEventListener('mouseup', () => mouseDown = false);

leftButton.addEventListener('mousedown', handler(2));
leftButton.addEventListener('mouseup', () => mouseDown = false);

rightButton.addEventListener('mousedown', handler(3));
rightButton.addEventListener('mouseup', () => mouseDown = false);

document.getElementById('train').addEventListener('click', () => train());
document.getElementById('predict').addEventListener('click', () => {
  startPacman();
  isPredicting = true;
  predict();
});

// Set hyper params from values above.
const learningRateElement =
    document.getElementById('learningRate') as HTMLInputElement;
learningRateElement.value = LEARNING_RATE.toString();
const getLearningRate = () => +learningRateElement.value;

const batchSizeFractionElement =
    document.getElementById('batchSizeFraction') as HTMLInputElement;
batchSizeFractionElement.value = BATCH_SIZE_FRACTION.toString();
const getBatchSizeFraction = () => +batchSizeFractionElement.value;

const epochsElement = document.getElementById('epochs') as HTMLInputElement;
epochsElement.value = '' + NUM_EPOCHS;
const getEpochs = () => +epochsElement.value;

const denseUnitsElement =
    document.getElementById('dense-units') as HTMLInputElement;
denseUnitsElement.value = DENSE_UNITS.toString();
const getDenseUnits = () => +denseUnitsElement.value;
const statusElement = document.getElementById('status');

async function pacman() {
  await webcam.setup();
  mobilenet = await loadMobilenet();

  // Show the controls once everything has loaded.
  document.getElementById('controls').style.display = '';
  (document.getElementsByClassName('train-container')[0] as HTMLDivElement)
      .style.visibility = 'visible';
  (document.getElementsByClassName('predict-container')[0] as HTMLDivElement)
      .style.visibility = 'visible';
  statusElement.style.visibility = 'hidden';
}

const pacmanElement = document.getElementById('pacman');

function startPacman() {
  fireEvent('N');
}
function fireEvent(keyCode: string) {
  const e = new KeyboardEvent('keydown');

  Object.defineProperty(e, 'keyCode', {
    get: () => {
      return KEY[keyCode];
    }
  });

  pacmanElement.dispatchEvent(e);
  document.dispatchEvent(e);
}

// TODO(nsthorat): Host this somewhere not here.
PACMAN.init(pacmanElement, '../node_modules/pacman/');

pacman();
