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

// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, BROWSER_ENVS, describeWithFlags, NODE_ENVS, registerTestEnv} from '@tensorflow/tfjs-core/dist/jasmine_util';

// Provide fake video stream
export function setupFakeVideoStream() {
  const width = 100;
  const height = 200;
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

export async function replaceHTMLVideoElementSource(
    videoElement: HTMLVideoElement) {
  const source = document.createElement('source');
  // tslint:disable:max-line-length
  source.src =
      'data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAu1tZGF0AAACrQYF//+p3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE1NSByMjkwMSA3ZDBmZjIyIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAxOCAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTMgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTEgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTI4LjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAAwZYiEAD//8m+P5OXfBeLGOfKE3xkODvFZuBflHv/+VwJIta6cbpIo4ABLoKBaYTkTAAAC7m1vb3YAAABsbXZoZAAAAAAAAAAAAAAAAAAAA+gAAAPoAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIYdHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAPoAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAACgAAAAWgAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAAD6AAAAAAAAQAAAAABkG1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAQAAAAEAAVcQAAAAAAC1oZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAATttaW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAD7c3RibAAAAJdzdHNkAAAAAAAAAAEAAACHYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAACgAFoASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAADFhdmNDAWQACv/hABhnZAAKrNlCjfkhAAADAAEAAAMAAg8SJZYBAAZo6+JLIsAAAAAYc3R0cwAAAAAAAAABAAAAAQAAQAAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAEAAAABAAAAFHN0c3oAAAAAAAAC5QAAAAEAAAAUc3RjbwAAAAAAAAABAAAAMAAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNTguMTIuMTAw';
  source.type = 'video/mp4';
  videoElement.srcObject = null;
  videoElement.appendChild(source);
  videoElement.play();

  if (videoElement.readyState < 2) {
    await new Promise(resolve => {
      videoElement.addEventListener('loadeddata', () => resolve());
    });
  }
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
