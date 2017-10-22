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

// tslint:disable-next-line:max-line-length
import {Array1D, Array2D, CheckpointLoader, NDArray, NDArrayMath, NDArrayMathGPU, Scalar} from '../deeplearn';
import * as demo_util from '../util';

import {KeyboardElement} from './keyboard_element';

// tslint:disable-next-line:no-require-imports
const Piano = require('tone-piano').Piano;

let lstmKernel1: Array2D;
let lstmBias1: Array1D;
let lstmKernel2: Array2D;
let lstmBias2: Array1D;
let lstmKernel3: Array2D;
let lstmBias3: Array1D;
let c: Array2D[];
let h: Array2D[];
let fullyConnectedBiases: Array1D;
let fullyConnectedWeights: Array2D;
const forgetBias = Scalar.new(1.0);
const activeNotes = new Map<number, number>();

// How many steps to generate per generateStep call.
// Generating more steps makes it less likely that we'll lag behind in note
// generation. Generating fewer steps makes it less likely that the browser UI
// thread will be starved for cycles.
const STEPS_PER_GENERATE_CALL = 10;
// How much time to try to generate ahead. More time means fewer buffer
// underruns, but also makes the lag from UI change to output larger.
const GENERATION_BUFFER_SECONDS = .5;
// If we're this far behind, reset currentTime time to piano.now().
const MAX_GENERATION_LAG_SECONDS = 1;
// If a note is held longer than this, release it.
const MAX_NOTE_DURATION_SECONDS = 3;

const NOTES_PER_OCTAVE = 12;
const DENSITY_BIN_RANGES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
const PITCH_HISTOGRAM_SIZE = NOTES_PER_OCTAVE;

let pitchHistogramEncoding: Array1D;
let noteDensityEncoding: Array1D;
let conditioningOff = true;

let currentTime = 0;
let currentVelocity = 100;

const MIN_MIDI_PITCH = 0;
const MAX_MIDI_PITCH = 127;
const VELOCITY_BINS = 32;
const MAX_SHIFT_STEPS = 100;
const STEPS_PER_SECOND = 100;

// The unique id of the currently scheduled setTimeout loop.
let currentLoopId = 0;

const EVENT_RANGES = [
  ['note_on', MIN_MIDI_PITCH, MAX_MIDI_PITCH],
  ['note_off', MIN_MIDI_PITCH, MAX_MIDI_PITCH],
  ['time_shift', 1, MAX_SHIFT_STEPS],
  ['velocity_change', 1, VELOCITY_BINS],
];

function calculateEventSize(): number {
  let eventOffset = 0;
  for (const eventRange of EVENT_RANGES) {
    const minValue = eventRange[1] as number;
    const maxValue = eventRange[2] as number;
    eventOffset += maxValue - minValue + 1;
  }
  return eventOffset;
}

const EVENT_SIZE = calculateEventSize();
const PRIMER_IDX = 355;  // shift 1s.
let lastSample = Scalar.new(PRIMER_IDX);

const container = document.querySelector('#keyboard');
const keyboardInterface = new KeyboardElement(container);

const piano = new Piano({velocities: 4}).toMaster();

const SALAMANDER_URL = 'https://storage.googleapis.com/learnjs-data/' +
    'Piano/Salamander/';
const CHECKPOINT_URL = 'https://storage.googleapis.com/learnjs-data/' +
    'checkpoint_zoo/performance_rnn';

const isDeviceSupported = demo_util.isWebGLSupported() && !demo_util.isSafari();

if (!isDeviceSupported) {
  document.querySelector('#status').innerHTML =
      'We do not yet support your device. Please try on a desktop ' +
      'computer with Chrome/Firefox, or an Android phone with WebGL support.';
} else {
  start();
}

const math = new NDArrayMathGPU();

function start() {
  piano.load(SALAMANDER_URL)
      .then(() => {
        const reader = new CheckpointLoader(CHECKPOINT_URL);
        return reader.getAllVariables();
      })
      .then((vars: {[varName: string]: NDArray}) => {
        document.querySelector('#status').classList.add('hidden');
        document.querySelector('#controls').classList.remove('hidden');
        document.querySelector('#keyboard').classList.remove('hidden');

        lstmKernel1 =
            vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'] as Array2D;
        lstmBias1 =
            vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'] as Array1D;

        lstmKernel2 =
            vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'] as Array2D;
        lstmBias2 =
            vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'] as Array1D;

        lstmKernel3 =
            vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/kernel'] as Array2D;
        lstmBias3 =
            vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/bias'] as Array1D;

        fullyConnectedBiases = vars['fully_connected/biases'] as Array1D;
        fullyConnectedWeights = vars['fully_connected/weights'] as Array2D;
        resetRnn();
      });
}

function resetRnn() {
  c = [
    Array2D.zeros([1, lstmBias1.shape[0] / 4]),
    Array2D.zeros([1, lstmBias2.shape[0] / 4]),
    Array2D.zeros([1, lstmBias3.shape[0] / 4]),
  ];
  h = [
    Array2D.zeros([1, lstmBias1.shape[0] / 4]),
    Array2D.zeros([1, lstmBias2.shape[0] / 4]),
    Array2D.zeros([1, lstmBias3.shape[0] / 4]),
  ];
  if (lastSample != null) {
    lastSample.dispose();
  }
  lastSample = Scalar.new(PRIMER_IDX);
  currentTime = piano.now();
  currentLoopId++;
  generateStep(currentLoopId);
}

window.addEventListener('resize', resize);

function resize() {
  keyboardInterface.resize();
}

resize();

const densityControl =
    document.getElementById('note-density') as HTMLInputElement;
const densityDisplay = document.getElementById('note-density-display');
const conditioningOffElem =
    document.getElementById('conditioning-off') as HTMLInputElement;
conditioningOffElem.onchange = updateConditioningParams;
const conditioningOnElem =
    document.getElementById('conditioning-on') as HTMLInputElement;
conditioningOnElem.onchange = updateConditioningParams;
const conditioningControlsElem =
    document.getElementById('conditioning-controls') as HTMLDivElement;

const pitchHistogramElements = [
  document.getElementById('pitch-c'),
  document.getElementById('pitch-cs'),
  document.getElementById('pitch-d'),
  document.getElementById('pitch-ds'),
  document.getElementById('pitch-e'),
  document.getElementById('pitch-f'),
  document.getElementById('pitch-fs'),
  document.getElementById('pitch-g'),
  document.getElementById('pitch-gs'),
  document.getElementById('pitch-a'),
  document.getElementById('pitch-as'),
  document.getElementById('pitch-b'),
] as HTMLInputElement[];

let preset1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
let preset2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

try {
  parseHash();
} catch (e) {
  // If we didn't successfully parse the hash, we can just use defaults.
  console.warn(e);
}

function parseHash() {
  if (!window.location.hash) {
    return;
  }
  const params = window.location.hash.substr(1).split('|');
  densityControl.value = params[0];
  const pitches = params[1].split(',');
  for (let i = 0; i < pitchHistogramElements.length; i++) {
    pitchHistogramElements[i].value = pitches[i];
  }
  const preset1Values = params[2].split(',');
  for (let i = 0; i < preset1.length; i++) {
    preset1[i] = parseInt(preset1Values[i], 10);
  }
  const preset2Values = params[3].split(',');
  for (let i = 0; i < preset2.length; i++) {
    preset2[i] = parseInt(preset2Values[i], 10);
  }
  if (!!parseInt(params[4], 10)) {
    conditioningOffElem.checked = true;
  } else {
    conditioningOnElem.checked = true;
  }
}

function updateConditioningParams() {
  const pitchHistogram = pitchHistogramElements.map((e) => {
    return parseInt(e.value, 10) || 0;
  });

  if (noteDensityEncoding !== undefined) {
    noteDensityEncoding.dispose();
    noteDensityEncoding = undefined;
  }

  if (conditioningOffElem.checked) {
    conditioningOff = true;
    conditioningControlsElem.classList.add('inactive');
  } else {
    conditioningOff = false;
    conditioningControlsElem.classList.remove('inactive');
  }

  window.location.assign(
      '#' + densityControl.value + '|' + pitchHistogram.join(',') + '|' +
      preset1.join(',') + '|' + preset2.join(',') + '|' +
      (conditioningOff ? '1' : '0'));

  const noteDensityIdx = parseInt(densityControl.value, 10) || 0;
  const noteDensity = DENSITY_BIN_RANGES[noteDensityIdx];
  densityDisplay.innerHTML = noteDensity.toString();
  noteDensityEncoding = Array1D.zeros([DENSITY_BIN_RANGES.length + 1]);
  noteDensityEncoding.set(1.0, noteDensityIdx + 1);

  if (pitchHistogramEncoding !== undefined) {
    pitchHistogramEncoding.dispose();
    pitchHistogramEncoding = undefined;
  }
  pitchHistogramEncoding = Array1D.zeros([PITCH_HISTOGRAM_SIZE]);
  const pitchHistogramTotal = pitchHistogram.reduce((prev, val) => {
    return prev + val;
  });
  for (let i = 0; i < PITCH_HISTOGRAM_SIZE; i++) {
    pitchHistogramEncoding.set(pitchHistogram[i] / pitchHistogramTotal, i);
  }
}

document.getElementById('note-density').oninput = updateConditioningParams;
pitchHistogramElements.map((e) => {
  e.oninput = updateConditioningParams;
});
updateConditioningParams();

function updatePitchHistogram(newHist: number[]) {
  for (let i = 0; i < newHist.length; i++) {
    pitchHistogramElements[i].value = newHist[i].toString();
  }
  updateConditioningParams();
}

document.getElementById('c-major').onclick = () => {
  updatePitchHistogram([2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]);
};

document.getElementById('f-major').onclick = () => {
  updatePitchHistogram([1, 0, 1, 0, 1, 2, 0, 1, 0, 1, 1, 0]);
};

document.getElementById('d-minor').onclick = () => {
  updatePitchHistogram([1, 0, 2, 0, 1, 1, 0, 1, 0, 1, 1, 0]);
};

document.getElementById('whole-tone').onclick = () => {
  updatePitchHistogram([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
};

document.getElementById('pentatonic').onclick = () => {
  updatePitchHistogram([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]);
};

document.getElementById('reset-rnn').onclick = () => {
  resetRnn();
};

document.getElementById('preset-1').onclick = () => {
  updatePitchHistogram(preset1);
};

document.getElementById('preset-2').onclick = () => {
  updatePitchHistogram(preset2);
};

document.getElementById('save-1').onclick = () => {
  preset1 = pitchHistogramElements.map((e) => {
    return parseInt(e.value, 10) || 0;
  });
  updateConditioningParams();
};

document.getElementById('save-2').onclick = () => {
  preset2 = pitchHistogramElements.map((e) => {
    return parseInt(e.value, 10) || 0;
  });
  updateConditioningParams();
};

function getConditioning(math: NDArrayMath): Array1D {
  return math.scope((keep, track) => {
    if (conditioningOff) {
      const size =
          1 + noteDensityEncoding.shape[0] + pitchHistogramEncoding.shape[0];
      const conditioning = track(Array1D.zeros([size]));
      conditioning.set(1.0, 0);
      return conditioning;
    } else {
      const conditioningValues =
          math.concat1D(noteDensityEncoding, pitchHistogramEncoding);
      return math.concat1D(
          track(Scalar.new(0.0).as1D()),  // conditioning on.
          conditioningValues);
    }
  });
}

async function generateStep(loopId: number) {
  if (loopId < currentLoopId) {
    // Was part of an outdated generateStep() scheduled via setTimeout.
    return;
  }
  await math.scope(async (keep, track) => {
    const lstm1 =
        math.basicLSTMCell.bind(math, forgetBias, lstmKernel1, lstmBias1);
    const lstm2 =
        math.basicLSTMCell.bind(math, forgetBias, lstmKernel2, lstmBias2);
    const lstm3 =
        math.basicLSTMCell.bind(math, forgetBias, lstmKernel3, lstmBias3);

    c.map(val => {
      track(val);
    });
    h.map(val => {
      track(val);
    });
    const outputs: Scalar[] = [];
    // Generate some notes.
    for (let i = 0; i < STEPS_PER_GENERATE_CALL; i++) {
      // Use last sampled output as the next input.
      const eventInput = math.oneHot(lastSample.as1D(), EVENT_SIZE).as1D();
      // Dispose the last sample from the previous generate call, since we
      // kept it.
      if (i === 0) {
        lastSample.dispose();
      }
      const conditioning = getConditioning(math);
      const input = math.concat1D(conditioning, eventInput);
      const output =
          math.multiRNNCell([lstm1, lstm2, lstm3], input.as2D(1, -1), c, h);
      c = output[0];
      h = output[1];

      const outputH = h[2];
      const weightedResult = math.matMul(outputH, fullyConnectedWeights);
      const logits = math.add(weightedResult, fullyConnectedBiases);

      const softmax = math.softmax(logits.as1D());
      const sampledOutput = math.multinomial(softmax, 1).asScalar();
      outputs.push(sampledOutput);
      keep(sampledOutput);
      lastSample = sampledOutput;
    }

    c.map(val => {
      keep(val);
    });
    h.map(val => {
      keep(val);
    });

    await outputs[outputs.length - 1].data();

    for (let i = 0; i < outputs.length; i++) {
      playOutput(await outputs[i].val());
    }

    // Pro-actively upload the last sample to the gpu again and keep it
    // for next time.
    lastSample.getTexture();

    if (piano.now() - currentTime > MAX_GENERATION_LAG_SECONDS) {
      console.warn(
          `Generation is ${piano.now() - currentTime} seconds behind, ` +
          `which is over ${MAX_NOTE_DURATION_SECONDS}. Resetting time!`);
      currentTime = piano.now();
    }
    const delta =
        Math.max(0, currentTime - piano.now() - GENERATION_BUFFER_SECONDS);
    setTimeout(() => generateStep(loopId), delta * 1000);
  });
}

/**
 * Decode the output index and play it on the piano and keyboardInterface.
 */
function playOutput(index: number) {
  let offset = 0;
  for (const eventRange of EVENT_RANGES) {
    const eventType = eventRange[0] as string;
    const minValue = eventRange[1] as number;
    const maxValue = eventRange[2] as number;
    if (offset <= index && index <= offset + maxValue - minValue) {
      if (eventType === 'note_on') {
        const noteNum = index - offset;
        setTimeout(() => {
          keyboardInterface.keyDown(noteNum);
          setTimeout(() => {
            keyboardInterface.keyUp(noteNum);
          }, 100);
        }, (currentTime - piano.now()) * 1000);
        activeNotes.set(noteNum, currentTime);
        return piano.keyDown(noteNum, currentTime, currentVelocity);
      } else if (eventType === 'note_off') {
        const noteNum = index - offset;
        piano.keyUp(
            noteNum, Math.max(currentTime, activeNotes.get(noteNum) + .5));
        activeNotes.delete(noteNum);
        return;
      } else if (eventType === 'time_shift') {
        currentTime += (index - offset + 1) / STEPS_PER_SECOND;
        activeNotes.forEach((time, noteNum) => {
          if (currentTime - time > MAX_NOTE_DURATION_SECONDS) {
            console.info(
                `Note ${noteNum} has been active for ${currentTime - time}, ` +
                `seconds which is over ${MAX_NOTE_DURATION_SECONDS}, will ` +
                `release.`);
            piano.keyUp(noteNum, currentTime);
            activeNotes.delete(noteNum);
          }
        });
        return currentTime;
      } else if (eventType === 'velocity_change') {
        currentVelocity = (index - offset + 1) * Math.ceil(127 / VELOCITY_BINS);
        currentVelocity = currentVelocity / 127;
        return currentVelocity;
      } else {
        throw new Error('Could not decode eventType: ' + eventType);
      }
    }
    offset += maxValue - minValue + 1;
  }
  throw new Error(`Could not decode index: ${index}`);
}
