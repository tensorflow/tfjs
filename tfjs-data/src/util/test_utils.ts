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

import {ALL_ENVS, BROWSER_ENVS, describeWithFlags, NODE_ENVS, registerTestEnv} from '@tensorflow/tfjs-core/dist/jasmine_util';

// Provide fake video stream
export function setupFakeVideoStream() {
  const width = 500;
  const height = 500;
  const canvasElement = document.createElement('canvas');
  const ctx = canvasElement.getContext('2d');
  ctx.fillStyle = 'rgb(1,2,3)';
  ctx.fillRect(0, 0, width, height);
  // tslint:disable-next-line:no-any
  const stream = (canvasElement as any).captureStream(60);
  navigator.mediaDevices.getUserMedia = async () => {
    return stream;
  };
}

// Register backends.
registerTestEnv({name: 'cpu', backendName: 'cpu'});
registerTestEnv({
  name: 'webgl2',
  backendName: 'webgl',
  flags: {
    'WEBGL_VERSION': 2,
    'WEBGL_CPU_FORWARD': false,
    'WEBGL_SIZE_UPLOAD_UNIFORM': 0
  }
});

export function describeAllEnvs(testName: string, tests: () => void) {
  describeWithFlags(testName, ALL_ENVS, () => {
    tests();
  });
}

export function describeBrowserEnvs(testName: string, tests: () => void) {
  describeWithFlags(testName, BROWSER_ENVS, () => {
    tests();
  });
}

export function describeNodeEnvs(testName: string, tests: () => void) {
  describeWithFlags(testName, NODE_ENVS, () => {
    tests();
  });
}

/**
 * Testing Utilities for browser audeo stream.
 */
export function setupFakeAudeoStream() {
  navigator.mediaDevices.getUserMedia = async () => {
    const stream = new MediaStream();
    return stream;
  };
  // tslint:disable-next-line:no-any
  (window as any).AudioContext = FakeAudioContext.createInstance;
}

export class FakeAudioContext {
  readonly sampleRate = 44100;

  static createInstance() {
    return new FakeAudioContext();
  }

  createMediaStreamSource() {
    return new FakeMediaStreamAudioSourceNode();
  }

  createAnalyser() {
    return new FakeAnalyser();
  }

  close(): void {}
}

export class FakeAudioMediaStream {
  constructor() {}
  getTracks(): Array<{}> {
    return [];
  }
}

class FakeMediaStreamAudioSourceNode {
  constructor() {}
  connect(node: {}): void {}
}

class FakeAnalyser {
  fftSize: number;
  smoothingTimeConstant: number;
  private x: number;
  constructor() {
    this.x = 0;
  }

  getFloatFrequencyData(data: Float32Array): void {
    const xs: number[] = [];
    for (let i = 0; i < this.fftSize / 2; ++i) {
      xs.push(this.x++);
    }
    data.set(new Float32Array(xs));
  }

  getFloatTimeDomainData(data: Float32Array): void {
    const xs: number[] = [];
    for (let i = 0; i < this.fftSize / 2; ++i) {
      xs.push(-(this.x++));
    }
    data.set(new Float32Array(xs));
  }

  disconnect(): void {}
}
