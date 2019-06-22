/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
 *
 * =============================================================================
 */

import {ENV, Tensor, tensor, Tensor2D, Tensor3D, TensorContainer, util} from '@tensorflow/tfjs-core';
import {MicrophoneConfig} from '../types';
import {LazyIterator} from './lazy_iterator';

/**
 * Provide a stream of tensors from microphone audio stream. The tensors are
 * representing audio data as frequency-domain spectrogram generated with
 * browser's native FFT. Tensors representing time-domain waveform is available
 * based on configuration. Only works in browser environment.
 */
export class MicrophoneIterator extends LazyIterator<TensorContainer> {
  private isClosed = false;
  private stream: MediaStream;
  private readonly fftSize: number;
  private readonly columnTruncateLength: number;
  private freqData: Float32Array;
  private timeData: Float32Array;
  private readonly numFrames: number;
  private analyser: AnalyserNode;
  private audioContext: AudioContext;
  private sampleRateHz: number;
  private readonly audioTrackConstraints: MediaTrackConstraints;
  private readonly smoothingTimeConstant: number;
  private readonly includeSpectrogram: boolean;
  private readonly includeWaveform: boolean;

  private constructor(protected readonly microphoneConfig: MicrophoneConfig) {
    super();
    this.fftSize = microphoneConfig.fftSize || 1024;
    const fftSizeLog2 = Math.log2(this.fftSize);
    if (this.fftSize < 0 || fftSizeLog2 < 4 || fftSizeLog2 > 14 ||
        !Number.isInteger(fftSizeLog2)) {
      throw new Error(
          `Invalid fftSize: it must be a power of 2 between ` +
          `2 to 4 and 2 to 14, but got ${this.fftSize}`);
    }

    this.numFrames = microphoneConfig.numFramesPerSpectrogram || 43;
    this.sampleRateHz = microphoneConfig.sampleRateHz;
    this.columnTruncateLength =
        microphoneConfig.columnTruncateLength || this.fftSize;
    this.audioTrackConstraints = microphoneConfig.audioTrackConstraints;
    this.smoothingTimeConstant = microphoneConfig.smoothingTimeConstant || 0;

    this.includeSpectrogram =
        microphoneConfig.includeSpectrogram === false ? false : true;
    this.includeWaveform =
        microphoneConfig.includeWaveform === true ? true : false;
    if (!this.includeSpectrogram && !this.includeWaveform) {
      throw new Error(
          'Both includeSpectrogram and includeWaveform are false. ' +
          'At least one type of data should be returned.');
    }
  }

  summary() {
    return `microphone`;
  }

  // Construct a MicrophoneIterator and start the audio stream.
  static async create(microphoneConfig: MicrophoneConfig = {}) {
    if (ENV.get('IS_NODE')) {
      throw new Error(
          'microphone API is only supported in browser environment.');
    }

    const microphoneIterator = new MicrophoneIterator(microphoneConfig);

    // Call async function start() to initialize the audio stream.
    await microphoneIterator.start();

    return microphoneIterator;
  }

  // Start the audio stream and FFT.
  async start(): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: this.audioTrackConstraints == null ? true :
                                                    this.audioTrackConstraints,
        video: false
      });
    } catch (e) {
      throw new Error(
          `Error thrown while initializing video stream: ${e.message}`);
    }

    if (!this.stream) {
      throw new Error('Could not obtain audio from microphone.');
    }

    const ctxConstructor =
        // tslint:disable-next-line:no-any
        (window as any).AudioContext || (window as any).webkitAudioContext;
    this.audioContext = new ctxConstructor();

    if (!this.sampleRateHz) {
      // If sample rate is not provided, use the available sample rate on
      // device.
      this.sampleRateHz = this.audioContext.sampleRate;
    } else if (this.audioContext.sampleRate !== this.sampleRateHz) {
      throw new Error(
          `Mismatch in sampling rate: ` +
          `Expected: ${this.sampleRateHz}; ` +
          `Actual: ${this.audioContext.sampleRate}`);
    }

    const streamSource = this.audioContext.createMediaStreamSource(this.stream);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = this.fftSize * 2;
    this.analyser.smoothingTimeConstant = this.smoothingTimeConstant;
    streamSource.connect(this.analyser);
    this.freqData = new Float32Array(this.fftSize);
    this.timeData = new Float32Array(this.fftSize);
    return;
  }

  async next(): Promise<IteratorResult<TensorContainer>> {
    if (this.isClosed) {
      return {value: null, done: true};
    }

    let spectrogramTensor: Tensor;
    let waveformTensor: Tensor;

    const audioDataQueue = await this.getAudioData();
    if (this.includeSpectrogram) {
      const freqData = this.flattenQueue(audioDataQueue.freqDataQueue);
      spectrogramTensor = this.getTensorFromAudioDataArray(
          freqData, [this.numFrames, this.columnTruncateLength, 1]);
    }
    if (this.includeWaveform) {
      const timeData = this.flattenQueue(audioDataQueue.timeDataQueue);
      waveformTensor = this.getTensorFromAudioDataArray(
          timeData, [this.numFrames * this.fftSize, 1]);
    }

    return {
      value: {'spectrogram': spectrogramTensor, 'waveform': waveformTensor},
      done: false
    };
  }

  // Capture one result from the audio stream, and extract the value from
  // iterator.next() result.
  async capture(): Promise<{spectrogram: Tensor3D, waveform: Tensor2D}> {
    return (await this.next()).value as
        {spectrogram: Tensor3D, waveform: Tensor2D};
  }

  private async getAudioData():
      Promise<{freqDataQueue: Float32Array[], timeDataQueue: Float32Array[]}> {
    const freqDataQueue: Float32Array[] = [];
    const timeDataQueue: Float32Array[] = [];
    let currentFrames = 0;
    return new Promise(resolve => {
      const intervalID = setInterval(() => {
        if (this.includeSpectrogram) {
          this.analyser.getFloatFrequencyData(this.freqData);
          // If the audio stream is initializing, return empty queue.
          if (this.freqData[0] === -Infinity) {
            resolve({freqDataQueue, timeDataQueue});
          }
          freqDataQueue.push(this.freqData.slice(0, this.columnTruncateLength));
        }
        if (this.includeWaveform) {
          this.analyser.getFloatTimeDomainData(this.timeData);
          timeDataQueue.push(this.timeData.slice());
        }

        // Clean interval and return when all frames have been collected
        if (++currentFrames === this.numFrames) {
          clearInterval(intervalID);
          resolve({freqDataQueue, timeDataQueue});
        }
      }, this.fftSize / this.sampleRateHz * 1e3);
    });
  }

  // Stop the audio stream and pause the iterator.
  stop(): void {
    this.isClosed = true;
    this.analyser.disconnect();
    this.audioContext.close();
    if (this.stream != null && this.stream.getTracks().length > 0) {
      this.stream.getTracks()[0].stop();
    }
  }

  // Override toArray() function to prevent collecting.
  toArray(): Promise<Tensor[]> {
    throw new Error('Can not convert infinite audio stream to array.');
  }

  // Return audio sampling rate in Hz
  getSampleRate(): number {
    return this.sampleRateHz;
  }

  private flattenQueue(queue: Float32Array[]): Float32Array {
    const frameSize = queue[0].length;
    const freqData = new Float32Array(queue.length * frameSize);
    queue.forEach((data, i) => freqData.set(data, i * frameSize));
    return freqData;
  }

  private getTensorFromAudioDataArray(freqData: Float32Array, shape: number[]):
      Tensor {
    const vals = new Float32Array(util.sizeFromShape(shape));
    // If the data is less than the output shape, the rest is padded with zeros.
    vals.set(freqData, vals.length - freqData.length);
    return tensor(vals, shape);
  }
}
