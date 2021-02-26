/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

import * as tf from '../index';
import {BROWSER_ENVS, describeWithFlags, NODE_ENVS} from '../jasmine_util';
import {expectArraysClose, expectArraysEqual} from '../test_util';

class MockContext {
  getImageData(x: number, y: number, width: number, height: number) {
    const data = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < data.length; ++i) {
      data[i] = i + 1;
    }
    return {data};
  }
}

class MockCanvas {
  constructor(public width: number, public height: number) {}
  getContext(type: '2d'): MockContext {
    return new MockContext();
  }
}

describeWithFlags('fromPixelsAsync, mock canvas', NODE_ENVS, () => {
  it('accepts a canvas-like element', async () => {
    const c = new MockCanvas(2, 2);
    // tslint:disable-next-line:no-any
    const t = await tf.browser.fromPixelsAsync(c as any);
    expect(t.dtype).toBe('int32');
    expect(t.shape).toEqual([2, 2, 3]);
    expectArraysEqual(
        await t.data(), [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]);
  });

  it('accepts a canvas-like element, numChannels=4', async () => {
    const c = new MockCanvas(2, 2);
    // tslint:disable-next-line:no-any
    const t = await tf.browser.fromPixelsAsync(c as any, 4);
    expect(t.dtype).toBe('int32');
    expect(t.shape).toEqual([2, 2, 4]);
    expectArraysEqual(
        await t.data(),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
  });
});

// Flag 'WRAP_TO_IMAGEBITMAP' is set by customer. The default
// value is false. The cases below won't try to wrap input to
// imageBitmap.
describeWithFlags('fromPixelsAsync, |WRAP_TO_IMAGEBITMAP| false',
  BROWSER_ENVS, () => {
  it('ImageData 1x1x3', async () => {
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 0;
    pixels.data[1] = 80;
    pixels.data[2] = 160;
    pixels.data[3] = 240;

    const array = await tf.browser.fromPixelsAsync(pixels, 3);

    expectArraysEqual(await array.data(), [0, 80, 160]);
  });

  it('ImageData 1x1x4', async () => {
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 0;
    pixels.data[1] = 80;
    pixels.data[2] = 160;
    pixels.data[3] = 240;

    const array = await tf.browser.fromPixelsAsync(pixels, 4);

    expectArraysEqual(await array.data(), [0, 80, 160, 240]);
  });

  it('ImageData 2x2x3', async () => {
    const pixels = new ImageData(2, 2);

    for (let i = 0; i < 8; i++) {
      pixels.data[i] = i * 2;
    }
    for (let i = 8; i < 16; i++) {
      pixels.data[i] = i * 2;
    }

    const array = await tf.browser.fromPixelsAsync(pixels, 3);

    expectArraysEqual(
        await array.data(), [0, 2, 4, 8, 10, 12, 16, 18, 20, 24, 26, 28]);
  });

  it('ImageData 2x2x4', async () => {
    const pixels = new ImageData(2, 2);
    for (let i = 0; i < 8; i++) {
      pixels.data[i] = i * 2;
    }
    for (let i = 8; i < 16; i++) {
      pixels.data[i] = i * 2;
    }

    const array = await tf.browser.fromPixelsAsync(pixels, 4);

    expectArraysClose(
        await array.data(),
        new Int32Array(
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]));
  });

  it('fromPixelsAsync, 3 channels', async () => {
    const pixels = new ImageData(1, 2);
    pixels.data[0] = 2;
    pixels.data[1] = 3;
    pixels.data[2] = 4;
    pixels.data[3] = 255;  // Not used.
    pixels.data[4] = 5;
    pixels.data[5] = 6;
    pixels.data[6] = 7;
    pixels.data[7] = 255;  // Not used.
    const res = await tf.browser.fromPixelsAsync(pixels, 3);
    expect(res.shape).toEqual([2, 1, 3]);
    expect(res.dtype).toBe('int32');
    expectArraysClose(await res.data(), [2, 3, 4, 5, 6, 7]);
  });

  it('fromPixelsAsync, reshape, then do tf.add()', async () => {
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 2;
    pixels.data[1] = 3;
    pixels.data[2] = 4;
    pixels.data[3] = 255;  // Not used.
    const content = await tf.browser.fromPixelsAsync(pixels, 3);
    const a = content.reshape([1, 1, 1, 3]);
    const res = a.add(tf.scalar(2, 'int32'));
    expect(res.shape).toEqual([1, 1, 1, 3]);
    expect(res.dtype).toBe('int32');
    expectArraysClose(await res.data(), [4, 5, 6]);
  });

  it('fromPixelsAsync + fromPixelsAsync', async () => {
    const pixelsA = new ImageData(1, 1);
    pixelsA.data[0] = 255;
    pixelsA.data[1] = 3;
    pixelsA.data[2] = 4;
    pixelsA.data[3] = 255;  // Not used.
    const pixelsB = new ImageData(1, 1);
    pixelsB.data[0] = 5;
    pixelsB.data[1] = 6;
    pixelsB.data[2] = 7;
    pixelsB.data[3] = 255;  // Not used.
    const contentA = await tf.browser.fromPixelsAsync(pixelsA, 3);
    const contentB = await tf.browser.fromPixelsAsync(pixelsB, 3);
    const a = contentA.toFloat();
    const b = contentB.toFloat();
    const res = a.add(b);
    expect(res.shape).toEqual([1, 1, 3]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [260, 9, 11]);
  });
  it('fromPixelsAsync for PixelData type', async () => {
    const dataA = new Uint8Array([255, 3, 4, 255]);
    const pixelsA = {width: 1, height: 1, data: dataA};

    const dataB = new Uint8Array([5, 6, 7, 255]);
    const pixelsB = {width: 1, height: 1, data: dataB};
    const contentA = await tf.browser.fromPixelsAsync(pixelsA, 3);
    const contentB = await tf.browser.fromPixelsAsync(pixelsB, 3);
    const a = contentA.toFloat();
    const b = contentB.toFloat();
    const res = a.add(b);
    expect(res.shape).toEqual([1, 1, 3]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [260, 9, 11]);
  });

  it('fromPixelsAsync for HTMLCanvasElement', async () => {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    const ctx = canvas.getContext('2d');
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 0;
    pixels.data[1] = 80;
    pixels.data[2] = 160;
    pixels.data[3] = 240;
    ctx.putImageData(pixels, 1, 1);
    const res = await tf.browser.fromPixelsAsync(canvas);
    expect(res.shape).toEqual([1, 1, 3]);
    const data = await res.data();
    expect(data.length).toEqual(1 * 1 * 3);
  });
  it('fromPixelsAsync for HTMLImageElement', async () => {
    const img = new Image(10, 10);
    img.src = 'data:image/gif;base64' +
        ',R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==';

    await new Promise(resolve => {
        img.onload = () => resolve(img);
    });

    const res = await tf.browser.fromPixelsAsync(img);
    expect(res.shape).toEqual([10, 10, 3]);
    const data = await res.data();
    expect(data.length).toEqual(10 * 10 * 3);
  });
  it('fromPixelsAsync for HTMLVideoElement', async () => {
    const video = document.createElement('video');
    video.autoplay = true;
    const source = document.createElement('source');
    // tslint:disable:max-line-length
    source.src =
        'data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAu1tZGF0AAACrQYF//+p3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE1NSByMjkwMSA3ZDBmZjIyIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAxOCAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTMgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTEgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTI4LjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAAwZYiEAD//8m+P5OXfBeLGOfKE3xkODvFZuBflHv/+VwJIta6cbpIo4ABLoKBaYTkTAAAC7m1vb3YAAABsbXZoZAAAAAAAAAAAAAAAAAAAA+gAAAPoAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIYdHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAPoAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAACgAAAAWgAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAAD6AAAAAAAAQAAAAABkG1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAQAAAAEAAVcQAAAAAAC1oZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAATttaW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAD7c3RibAAAAJdzdHNkAAAAAAAAAAEAAACHYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAACgAFoASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAADFhdmNDAWQACv/hABhnZAAKrNlCjfkhAAADAAEAAAMAAg8SJZYBAAZo6+JLIsAAAAAYc3R0cwAAAAAAAAABAAAAAQAAQAAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAEAAAABAAAAFHN0c3oAAAAAAAAC5QAAAAEAAAAUc3RjbwAAAAAAAAABAAAAMAAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNTguMTIuMTAw';
    source.type = 'video/mp4';
    video.appendChild(source);
    document.body.appendChild(video);

    // On mobile safari the ready state is ready immediately so we
    if (video.readyState < 2) {
      await new Promise(resolve => {
        video.addEventListener('loadeddata', () => resolve(video));
      });
    }

    const res = await tf.browser.fromPixelsAsync(video);
    expect(res.shape).toEqual([90, 160, 3]);
    const data = await res.data();
    expect(data.length).toEqual(90 * 160 * 3);
    document.body.removeChild(video);
  });

  it('canvas and image match', async () => {
    const img = new Image();
    const size = 80;
    // tslint:disable:max-line-length
    img.src =
        'data:image/jpeg;base64,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';
    // tslint:enable:max-line-length

    await new Promise(resolve => {
      img.onload = () => resolve(img);
    });

    img.width = size;
    img.height = size;

    const pixels = await tf.browser.fromPixelsAsync(img, 4);

    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, size, size);
    const actual = ctx.getImageData(0, 0, size, size).data;
    const actualInt32 = Int32Array.from(actual);
    const pixelsData = await pixels.data();

    expectArraysClose(pixelsData, actualInt32, 10);
  });
});

// Flag |WRAP_TO_IMAGEBITMAP| is set by customer.
// This flag helps on image, video and canvas input cases
// for WebGPU backends. We'll cover these inputs with test
// cases set 'WRAP_TO_IMAGEBITMAP' to true.
describeWithFlags('fromPixelsAsync, |WRAP_TO_IMAGEBITMAP| true',
  BROWSER_ENVS, () => {
  beforeAll(() => {
      tf.env().set('WRAP_TO_IMAGEBITMAP', true);
  });

  afterAll(() => {
    tf.env().set('WRAP_TO_IMAGEBITMAP', false);
  });

  it('fromPixelsAsync for HTMLCanvasElement ', async () => {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    const ctx = canvas.getContext('2d');
    const pixels = new ImageData(1, 1);
    pixels.data[0] = 0;
    pixels.data[1] = 80;
    pixels.data[2] = 160;
    pixels.data[3] = 240;
    ctx.putImageData(pixels, 1, 1);
    const res = await tf.browser.fromPixelsAsync(canvas);
    expect(res.shape).toEqual([1, 1, 3]);
    const data = await res.data();
    expect(data.length).toEqual(1 * 1 * 3);
  });
  it('fromPixelsAsync for HTMLImageElement', async () => {
    const img = new Image(10, 10);
    img.src = 'data:image/gif;base64' +
        ',R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==';

    await new Promise(resolve => {
        img.onload = () => resolve(img);
    });

    const res = await tf.browser.fromPixelsAsync(img);
    expect(res.shape).toEqual([10, 10, 3]);
    const data = await res.data();
    expect(data.length).toEqual(10 * 10 * 3);
  });
  it('fromPixelsAsync for HTMLVideoElement', async () => {
    const video = document.createElement('video');
    video.autoplay = true;
    const source = document.createElement('source');
    // tslint:disable:max-line-length
    source.src =
        'data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAu1tZGF0AAACrQYF//+p3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE1NSByMjkwMSA3ZDBmZjIyIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAxOCAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTMgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTEgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTI4LjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAAwZYiEAD//8m+P5OXfBeLGOfKE3xkODvFZuBflHv/+VwJIta6cbpIo4ABLoKBaYTkTAAAC7m1vb3YAAABsbXZoZAAAAAAAAAAAAAAAAAAAA+gAAAPoAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIYdHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAPoAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAACgAAAAWgAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAAD6AAAAAAAAQAAAAABkG1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAQAAAAEAAVcQAAAAAAC1oZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAATttaW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAD7c3RibAAAAJdzdHNkAAAAAAAAAAEAAACHYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAACgAFoASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAADFhdmNDAWQACv/hABhnZAAKrNlCjfkhAAADAAEAAAMAAg8SJZYBAAZo6+JLIsAAAAAYc3R0cwAAAAAAAAABAAAAAQAAQAAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAEAAAABAAAAFHN0c3oAAAAAAAAC5QAAAAEAAAAUc3RjbwAAAAAAAAABAAAAMAAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNTguMTIuMTAw';
    source.type = 'video/mp4';
    video.appendChild(source);
    document.body.appendChild(video);

    // On mobile safari the ready state is ready immediately so we
    if (video.readyState < 2) {
      await new Promise(resolve => {
        video.addEventListener('loadeddata', () => resolve(video));
      });
    }

    const res = await tf.browser.fromPixelsAsync(video);
    expect(res.shape).toEqual([90, 160, 3]);
    const data = await res.data();
    expect(data.length).toEqual(90 * 160 * 3);
    document.body.removeChild(video);
  });

  it('canvas and image match', async () => {
    const img = new Image();
    const size = 80;
    // tslint:disable:max-line-length
    img.src =
        'data:image/jpeg;base64,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';
    // tslint:enable:max-line-length

    await new Promise(resolve => {
      img.onload = () => resolve(img);
    });

    img.width = size;
    img.height = size;

    const pixels = await tf.browser.fromPixelsAsync(img, 4);

    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, size, size);
    const actual = ctx.getImageData(0, 0, size, size).data;
    const actualInt32 = Int32Array.from(actual);
    const pixelsData = await pixels.data();

    expectArraysClose(pixelsData, actualInt32, 10);
  });
});
