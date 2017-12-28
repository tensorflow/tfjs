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
import {Array1D, Array2D, CheckpointLoader, ENV, NDArray, NDArrayMath, Scalar} from 'deeplearn';
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

const RESET_RNN_FREQUENCY_MS = 30000;

let pitchHistogramEncoding: Array1D;
let noteDensityEncoding: Array1D;
let conditioned = false;

let currentPianoTimeSec = 0;
// When the piano roll starts in browser-time via performance.now().
let pianoStartTimestampMs = 0;

let currentVelocity = 100;

const MIN_MIDI_PITCH = 0;
const MAX_MIDI_PITCH = 127;
const VELOCITY_BINS = 32;
const MAX_SHIFT_STEPS = 100;
const STEPS_PER_SECOND = 100;

const MIDI_EVENT_ON = 0x90;
const MIDI_EVENT_OFF = 0x80;
const MIDI_NO_OUTPUT_DEVICES_FOUND_MESSAGE = 'No midi output devices found.';
const MIDI_NO_INPUT_DEVICES_FOUND_MESSAGE = 'No midi input devices found.';

const MID_IN_CHORD_RESET_THRESHOLD_MS = 1000;

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
let lastSample = Scalar.new(PRIMER_IDX, 'int32');

const container = document.querySelector('#keyboard');
const keyboardInterface = new KeyboardElement(container);

const piano = new Piano({velocities: 4}).toMaster();

const SALAMANDER_URL = 'https://storage.googleapis.com/learnjs-data/' +
    'Piano/Salamander/';
const CHECKPOINT_URL = 'https://storage.googleapis.com/learnjs-data/' +
    'checkpoint_zoo/performance_rnn_v2';

const isDeviceSupported = demo_util.isWebGLSupported();

if (!isDeviceSupported) {
  document.querySelector('#status').innerHTML =
      'We do not yet support your device. Please try on a desktop ' +
      'computer with Chrome/Firefox, or an Android phone with WebGL support.';
} else {
  start();
}

const math = ENV.math;

let modelReady = false;

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
        modelReady = true;
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
  currentPianoTimeSec = piano.now();
  pianoStartTimestampMs = performance.now() - currentPianoTimeSec * 1000;
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
conditioningOffElem.onchange = disableConditioning;
const conditioningOnElem =
    document.getElementById('conditioning-on') as HTMLInputElement;
conditioningOnElem.onchange = enableConditioning;
setTimeout(() => disableConditioning());

const conditioningControlsElem =
    document.getElementById('conditioning-controls') as HTMLDivElement;

const gainSliderElement = document.getElementById('gain') as HTMLInputElement;
const gainDisplayElement =
    document.getElementById('gain-display') as HTMLSpanElement;
let globalGain = +gainSliderElement.value;
gainDisplayElement.innerText = globalGain.toString();
gainSliderElement.addEventListener('input', () => {
  globalGain = +gainSliderElement.value;
  gainDisplayElement.innerText = globalGain.toString();
});

const notes = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b'];

const pitchHistogramElements = notes.map(
    note => document.getElementById('pitch-' + note) as HTMLInputElement);
const histogramDisplayElements = notes.map(
    note => document.getElementById('hist-' + note) as HTMLDivElement);

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
  if (params[4] === 'true') {
    enableConditioning();

  } else if (params[4] === 'false') {
    disableConditioning();
  }
}

function enableConditioning() {
  conditioned = true;
  conditioningOffElem.checked = false;
  conditioningOnElem.checked = true;

  conditioningControlsElem.classList.remove('inactive');
  conditioningControlsElem.classList.remove('midicondition');

  updateConditioningParams();
}
function disableConditioning() {
  conditioned = false;
  conditioningOffElem.checked = true;
  conditioningOnElem.checked = false;

  conditioningControlsElem.classList.add('inactive');
  conditioningControlsElem.classList.remove('midicondition');

  updateConditioningParams();
}

function updateConditioningParams() {
  const pitchHistogram = pitchHistogramElements.map(e => {
    return parseInt(e.value, 10) || 0;
  });
  updateDisplayHistogram(pitchHistogram);

  if (noteDensityEncoding != null) {
    noteDensityEncoding.dispose();
    noteDensityEncoding = null;
  }

  window.location.assign(
      '#' + densityControl.value + '|' + pitchHistogram.join(',') + '|' +
      preset1.join(',') + '|' + preset2.join(',') + '|' +
      (conditioned ? 'true' : 'false'));

  const noteDensityIdx = parseInt(densityControl.value, 10) || 0;
  const noteDensity = DENSITY_BIN_RANGES[noteDensityIdx];
  densityDisplay.innerHTML = noteDensity.toString();
  noteDensityEncoding = Array1D.zeros([DENSITY_BIN_RANGES.length + 1]);
  noteDensityEncoding.set(1.0, noteDensityIdx + 1);

  if (pitchHistogramEncoding != null) {
    pitchHistogramEncoding.dispose();
    pitchHistogramEncoding = null;
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
pitchHistogramElements.forEach(e => {
  e.oninput = updateConditioningParams;
});
updateConditioningParams();

function updatePitchHistogram(newHist: number[]) {
  let allZero = true;
  for (let i = 0; i < newHist.length; i++) {
    allZero = allZero && newHist[i] === 0;
  }
  if (allZero) {
    newHist = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  }
  for (let i = 0; i < newHist.length; i++) {
    pitchHistogramElements[i].value = newHist[i].toString();
  }

  updateConditioningParams();
}
function updateDisplayHistogram(hist: number[]) {
  let sum = 0;
  for (let i = 0; i < hist.length; i++) {
    sum += hist[i];
  }

  for (let i = 0; i < hist.length; i++) {
    histogramDisplayElements[i].style.height =
        (100 * (hist[i] / sum)).toString() + 'px';
  }
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
  return math.scope(keep => {
    if (!conditioned) {
      // TODO(nsthorat): figure out why we have to cast these shapes to numbers.
      // The linter is complaining, though VSCode can infer the types.
      const size = 1 + (noteDensityEncoding.shape[0] as number) +
          (pitchHistogramEncoding.shape[0] as number);
      const conditioning = Array1D.zeros([size]);
      conditioning.set(1.0, 0);
      return conditioning;
    } else {
      const conditioningValues =
          math.concat1D(noteDensityEncoding, pitchHistogramEncoding);
      return math.concat1D(
          Scalar.new(0.0).as1D(),  // conditioning on.
          conditioningValues);
    }
  });
}

async function generateStep(loopId: number) {
  if (loopId < currentLoopId) {
    // Was part of an outdated generateStep() scheduled via setTimeout.
    return;
  }
  await math.scope(async keep => {
    const lstm1 =
        math.basicLSTMCell.bind(math, forgetBias, lstmKernel1, lstmBias1);
    const lstm2 =
        math.basicLSTMCell.bind(math, forgetBias, lstmKernel2, lstmBias2);
    const lstm3 =
        math.basicLSTMCell.bind(math, forgetBias, lstmKernel3, lstmBias3);

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

    c.forEach(val => keep(val));
    h.forEach(val => keep(val));

    await outputs[outputs.length - 1].data();

    for (let i = 0; i < outputs.length; i++) {
      playOutput(await outputs[i].val());
    }

    if (piano.now() - currentPianoTimeSec > MAX_GENERATION_LAG_SECONDS) {
      console.warn(
          `Generation is ${
              piano.now() - currentPianoTimeSec} seconds behind, ` +
          `which is over ${MAX_NOTE_DURATION_SECONDS}. Resetting time!`);
      currentPianoTimeSec = piano.now();
    }
    const delta = Math.max(
        0, currentPianoTimeSec - piano.now() - GENERATION_BUFFER_SECONDS);
    setTimeout(() => generateStep(loopId), delta * 1000);
  });
}

let midi;
// tslint:disable-next-line:no-any
let activeMidiOutputDevice: any = null;
// tslint:disable-next-line:no-any
let activeMidiInputDevice: any = null;
(async () => {
  const midiOutDropdownContainer =
      document.getElementById('midi-out-container');
  const midiInDropdownContainer = document.getElementById('midi-in-container');
  try {
    // tslint:disable-next-line:no-any
    const navigator: any = window.navigator;
    midi = await navigator.requestMIDIAccess();

    const midiOutDropdown =
        document.getElementById('midi-out') as HTMLSelectElement;
    const midiInDropdown =
        document.getElementById('midi-in') as HTMLSelectElement;

    let outputDeviceCount = 0;
    // tslint:disable-next-line:no-any
    const midiOutputDevices: any[] = [];
    // tslint:disable-next-line:no-any
    midi.outputs.forEach((output: any) => {
      console.log(`
          Output midi device [type: '${output.type}']
          id: ${output.id}
          manufacturer: ${output.manufacturer}
          name:${output.name}
          version: ${output.version}`);
      midiOutputDevices.push(output);

      const option = document.createElement('option');
      option.innerText = output.name;
      midiOutDropdown.appendChild(option);
      outputDeviceCount++;
    });

    midiOutDropdown.addEventListener('change', () => {
      activeMidiOutputDevice =
          midiOutputDevices[midiOutDropdown.selectedIndex - 1];
    });

    if (outputDeviceCount === 0) {
      midiOutDropdownContainer.innerText = MIDI_NO_OUTPUT_DEVICES_FOUND_MESSAGE;
    }

    let inputDeviceCount = 0;
    // tslint:disable-next-line:no-any
    const midiInputDevices: any[] = [];
    // tslint:disable-next-line:no-any
    midi.inputs.forEach((input: any) => {
      console.log(`
        Input midi device [type: '${input.type}']
        id: ${input.id}
        manufacturer: ${input.manufacturer}
        name:${input.name}
        version: ${input.version}`);
      midiInputDevices.push(input);

      const option = document.createElement('option');
      option.innerText = input.name;
      midiInDropdown.appendChild(option);
      inputDeviceCount++;
    });

    // tslint:disable-next-line:no-any
    const setActiveMidiInputDevice = (device: any) => {
      if (activeMidiInputDevice != null) {
        activeMidiInputDevice.onmidimessage = () => {};
      }
      activeMidiInputDevice = device;
      // tslint:disable-next-line:no-any
      device.onmidimessage = (event: any) => {
        const data = event.data;
        const type = data[0] & 0xf0;
        const note = data[1];
        const velocity = data[2];
        if (type === 144) {
          midiInNoteOn(note, velocity);
        }
      };
    };
    midiInDropdown.addEventListener('change', () => {
      setActiveMidiInputDevice(
          midiInputDevices[midiInDropdown.selectedIndex - 1]);
    });
    if (inputDeviceCount === 0) {
      midiInDropdownContainer.innerText = MIDI_NO_INPUT_DEVICES_FOUND_MESSAGE;
    }
  } catch (e) {
    midiOutDropdownContainer.innerText = MIDI_NO_OUTPUT_DEVICES_FOUND_MESSAGE;

    midi = null;
  }
})();

/**
 * Handle midi input.
 */
const CONDITIONING_OFF_TIME_MS = 30000;
let lastNotePressedTime = performance.now();
let midiInPitchHistogram = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
function midiInNoteOn(midiNote: number, velocity: number) {
  const now = performance.now();
  if (now - lastNotePressedTime > MID_IN_CHORD_RESET_THRESHOLD_MS) {
    midiInPitchHistogram = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    resetRnn();
  }
  lastNotePressedTime = now;

  // Turn on conditioning when a note is pressed/
  if (!conditioned) {
    resetRnn();
    enableConditioning();
  }

  // Turn off conditioning after 30 seconds unless other notes have been played.
  setTimeout(() => {
    if (performance.now() - lastNotePressedTime > CONDITIONING_OFF_TIME_MS) {
      disableConditioning();
      resetRnn();
    }
  }, CONDITIONING_OFF_TIME_MS);

  const note = midiNote % 12;
  midiInPitchHistogram[note]++;

  updateMidiInConditioning();
}

function updateMidiInConditioning() {
  updatePitchHistogram(midiInPitchHistogram);
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
        }, (currentPianoTimeSec - piano.now()) * 1000);
        activeNotes.set(noteNum, currentPianoTimeSec);

        if (activeMidiOutputDevice != null) {
          try {
            activeMidiOutputDevice.send(
                [
                  MIDI_EVENT_ON, noteNum,
                  Math.min(Math.floor(currentVelocity * globalGain), 127)
                ],
                Math.floor(1000 * currentPianoTimeSec) - pianoStartTimestampMs);
          } catch (e) {
            console.log(
                'Error sending midi note on event to midi output device:');
            console.log(e);
          }
        }

        return piano.keyDown(
            noteNum, currentPianoTimeSec, currentVelocity * globalGain / 100);
      } else if (eventType === 'note_off') {
        const noteNum = index - offset;

        const activeNoteEndTimeSec = activeNotes.get(noteNum);
        // If the note off event is generated for a note that hasn't been
        // pressed, just ignore it.
        if (activeNoteEndTimeSec == null) {
          return;
        }
        const timeSec =
            Math.max(currentPianoTimeSec, activeNoteEndTimeSec + .5);

        if (activeMidiOutputDevice != null) {
          activeMidiOutputDevice.send(
              [
                MIDI_EVENT_OFF, noteNum,
                Math.min(Math.floor(currentVelocity * globalGain), 127)
              ],
              Math.floor(timeSec * 1000) - pianoStartTimestampMs);
        }
        piano.keyUp(noteNum, timeSec);
        activeNotes.delete(noteNum);
        return;
      } else if (eventType === 'time_shift') {
        currentPianoTimeSec += (index - offset + 1) / STEPS_PER_SECOND;
        activeNotes.forEach((timeSec, noteNum) => {
          if (currentPianoTimeSec - timeSec > MAX_NOTE_DURATION_SECONDS) {
            console.info(
                `Note ${noteNum} has been active for ${
                    currentPianoTimeSec - timeSec}, ` +
                `seconds which is over ${MAX_NOTE_DURATION_SECONDS}, will ` +
                `release.`);
            if (activeMidiOutputDevice != null) {
              activeMidiOutputDevice.send([
                MIDI_EVENT_OFF, noteNum,
                Math.min(Math.floor(currentVelocity * globalGain), 127)
              ]);
            }
            piano.keyUp(noteNum, currentPianoTimeSec);
            activeNotes.delete(noteNum);
          }
        });
        return currentPianoTimeSec;
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

// Reset the RNN repeatedly so it doesn't trail off into incoherent musical
// babble.
const resettingText = document.getElementById('resettingText');
function resetRnnRepeatedly() {
  if (modelReady) {
    resetRnn();
    resettingText.style.opacity = '100';
  }

  setTimeout(() => {
    resettingText.style.opacity = '0';
  }, 1000);
  setTimeout(resetRnnRepeatedly, RESET_RNN_FREQUENCY_MS);
}
setTimeout(resetRnnRepeatedly, RESET_RNN_FREQUENCY_MS);
