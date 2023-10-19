/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';
import {WebGPUBackend} from './backend_webgpu';
import {describeWebGPU} from './test_util';

describeWebGPU('fromPixels', () => {
  let originalTimeout: number;
  beforeAll(() => {
    originalTimeout = jasmine.DEFAULT_TIMEOUT_INTERVAL;
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });
  afterAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = originalTimeout;
  });

  // Device is lost on Linux
  // tslint:disable-next-line: ban
  xit('should behave well if WEBGPU_IMPORT_EXTERNAL_TEXTURE is true or false',
      async () => {
        const oldImportExternalTexture =
            tf.env().getBool('WEBGPU_IMPORT_EXTERNAL_TEXTURE');
        const backend = tf.backend() as WebGPUBackend;
        const textureManager = backend.textureManager;
        textureManager.dispose();

        const source = document.createElement('source');
        source.src =
            // tslint:disable-next-line:max-line-length
            'data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAu1tZGF0AAACrQYF//+p3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE1NSByMjkwMSA3ZDBmZjIyIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAxOCAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbCAtIG9wdGlvbnM6IGNhYmFjPTEgcmVmPTMgZGVibG9jaz0xOjA6MCBhbmFseXNlPTB4MzoweDExMyBtZT1oZXggc3VibWU9NyBwc3k9MSBwc3lfcmQ9MS4wMDowLjAwIG1peGVkX3JlZj0xIG1lX3JhbmdlPTE2IGNocm9tYV9tZT0xIHRyZWxsaXM9MSA4eDhkY3Q9MSBjcW09MCBkZWFkem9uZT0yMSwxMSBmYXN0X3Bza2lwPTEgY2hyb21hX3FwX29mZnNldD0tMiB0aHJlYWRzPTMgbG9va2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxhY2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHlyYW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTEgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTI4LjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAAwZYiEAD//8m+P5OXfBeLGOfKE3xkODvFZuBflHv/+VwJIta6cbpIo4ABLoKBaYTkTAAAC7m1vb3YAAABsbXZoZAAAAAAAAAAAAAAAAAAAA+gAAAPoAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAIYdHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAPoAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAACgAAAAWgAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAAD6AAAAAAAAQAAAAABkG1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAQAAAAEAAVcQAAAAAAC1oZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAATttaW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAD7c3RibAAAAJdzdHNkAAAAAAAAAAEAAACHYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAACgAFoASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAADFhdmNDAWQACv/hABhnZAAKrNlCjfkhAAADAAEAAAMAAg8SJZYBAAZo6+JLIsAAAAAYc3R0cwAAAAAAAAABAAAAAQAAQAAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAEAAAABAAAAFHN0c3oAAAAAAAAC5QAAAAEAAAAUc3RjbwAAAAAAAAABAAAAMAAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNTguMTIuMTAw';
        source.type = 'video/mp4';

        const video = await test_util.createVideoElement(source);
        document.body.appendChild(video);
        await test_util.play(video);

        {
          tf.env().set('WEBGPU_IMPORT_EXTERNAL_TEXTURE', true);
          const res = tf.browser.fromPixels(video);
          expect(res.shape).toEqual([90, 160, 3]);
          const data = await res.data();
          expect(data.length).toEqual(90 * 160 * 3);
          const freeTexturesAfterFromPixels =
              textureManager.getNumFreeTextures();
          expect(freeTexturesAfterFromPixels).toEqual(0);
          const usedTexturesAfterFromPixels =
              textureManager.getNumUsedTextures();
          expect(usedTexturesAfterFromPixels).toEqual(0);
        }

        {
          tf.env().set('WEBGPU_IMPORT_EXTERNAL_TEXTURE', false);
          const res = tf.browser.fromPixels(video);
          expect(res.shape).toEqual([90, 160, 3]);
          const data = await res.data();
          expect(data.length).toEqual(90 * 160 * 3);
          const freeTexturesAfterFromPixels =
              textureManager.getNumFreeTextures();
          expect(freeTexturesAfterFromPixels).toEqual(1);
          const usedTexturesAfterFromPixels =
              textureManager.getNumUsedTextures();
          expect(usedTexturesAfterFromPixels).toEqual(0);
        }

        document.body.removeChild(video);
        tf.env().set(
            'WEBGPU_IMPORT_EXTERNAL_TEXTURE', oldImportExternalTexture);
      });

  // Failing on Linux
  // tslint:disable-next-line: ban
  xit('should reuse texture when fromPixels have same input size', async () => {
    const backend = tf.backend() as WebGPUBackend;
    const textureManager = backend.textureManager;
    textureManager.dispose();

    {
      const img = new Image(10, 10);
      img.src = 'data:image/gif;base64' +
          ',R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==';

      await new Promise(resolve => {
        img.onload = () => resolve(img);
      });

      const resImage = tf.browser.fromPixels(img);
      expect(resImage.shape).toEqual([10, 10, 3]);

      const dataImage = await resImage.data();
      expect(dataImage[0]).toEqual(0);
      expect(dataImage.length).toEqual(10 * 10 * 3);
      const freeTexturesAfterFromPixels = textureManager.getNumFreeTextures();
      expect(freeTexturesAfterFromPixels).toEqual(1);
      const usedTexturesAfterFromPixels = textureManager.getNumUsedTextures();
      expect(usedTexturesAfterFromPixels).toEqual(0);
    }

    {
      const img = new Image(10, 10);
      img.src =
          'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAABfSURBVChTY/gPBu8NLd/KqLxT1oZw4QAqDZSDoPeWDj9WrYUIAgG6NBAhm4FFGoIgxuCUBiKgMfikv1bW4pQGav334wdUGshBk/6SVQAUh0p/mzIDTQ6oFSGNHfz/DwAwi8mNzTi6rwAAAABJRU5ErkJggg==';
      await new Promise(resolve => {
        img.onload = () => resolve(img);
      });
      const resImage = tf.browser.fromPixels(img);
      expect(resImage.shape).toEqual([10, 10, 3]);

      const dataImage = await resImage.data();
      expect(dataImage[0]).toEqual(255);
      expect(dataImage.length).toEqual(10 * 10 * 3);
      const freeTexturesAfterFromPixels = textureManager.getNumFreeTextures();
      expect(freeTexturesAfterFromPixels).toEqual(1);
      const usedTexturesAfterFromPixels = textureManager.getNumUsedTextures();
      expect(usedTexturesAfterFromPixels).toEqual(0);
    }
  });
});
