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
 * =============================================================================
 */

import {env} from '../environment';
import {BROWSER_ENVS, describeWithFlags} from '../jasmine_util';

import {PlatformBrowser} from './platform_browser';

describeWithFlags('PlatformBrowser', BROWSER_ENVS, async () => {
  it('fetch calls window.fetch', async () => {
    const response = new Response();
    spyOn(self, 'fetch').and.returnValue(Promise.resolve(response));
    const platform = new PlatformBrowser();

    await platform.fetch('test/url', {method: 'GET'});

    expect(self.fetch).toHaveBeenCalledWith('test/url', {method: 'GET'});
  });

  it('now should use performance.now', async () => {
    const platform = new PlatformBrowser();

    const ms = 1234567;
    spyOn(performance, 'now').and.returnValue(ms);
    expect(platform.now()).toEqual(ms);
  });

  it('encodeUTF8 single string', () => {
    const platform = new PlatformBrowser();
    const bytes = platform.encode('hello', 'utf-8');
    expect(bytes.length).toBe(5);
    expect(bytes).toEqual(new Uint8Array([104, 101, 108, 108, 111]));
  });

  it('encodeUTF8 two strings delimited', () => {
    const platform = new PlatformBrowser();
    const bytes = platform.encode('hello\x00world', 'utf-8');
    expect(bytes.length).toBe(11);
    expect(bytes).toEqual(
        new Uint8Array([104, 101, 108, 108, 111, 0, 119, 111, 114, 108, 100]));
  });

  it('encodeUTF8 cyrillic', () => {
    const platform = new PlatformBrowser();
    const bytes = platform.encode('Здраво', 'utf-8');
    expect(bytes.length).toBe(12);
    expect(bytes).toEqual(new Uint8Array(
        [208, 151, 208, 180, 209, 128, 208, 176, 208, 178, 208, 190]));
  });

  it('decode single string', () => {
    const platform = new PlatformBrowser();
    const s =
        platform.decode(new Uint8Array([104, 101, 108, 108, 111]), 'utf-8');
    expect(s.length).toBe(5);
    expect(s).toEqual('hello');
  });

  it('decode two strings delimited', () => {
    const platform = new PlatformBrowser();
    const s = platform.decode(
        new Uint8Array([104, 101, 108, 108, 111, 0, 119, 111, 114, 108, 100]),
        'utf-8');
    expect(s.length).toBe(11);
    expect(s).toEqual('hello\x00world');
  });

  it('decode cyrillic', () => {
    const platform = new PlatformBrowser();
    const s = platform.decode(
        new Uint8Array(
            [208, 151, 208, 180, 209, 128, 208, 176, 208, 178, 208, 190]),
        'utf-8');
    expect(s.length).toBe(6);
    expect(s).toEqual('Здраво');
  });
});

describeWithFlags('setTimeout', BROWSER_ENVS, () => {
  const totalCount = 100;
  // Skip the first few samples because the browser does not clamp the timeout
  const skipCount = 5;

  it('setTimeout', (done) => {
    let count = 0;
    let startTime = performance.now();
    let totalTime = 0;
    setTimeout(_testSetTimeout, 0);

    function _testSetTimeout() {
      const endTime = performance.now();
      count++;
      if (count > skipCount) {
        totalTime += endTime - startTime;
      }
      if (count === totalCount) {
        const averageTime = totalTime / (totalCount - skipCount);
        console.log(`averageTime of setTimeout is ${averageTime} ms`);
        expect(averageTime).toBeGreaterThan(4);
        done();
        return;
      }
      startTime = performance.now();
      setTimeout(_testSetTimeout, 0);
    }
  });

  it('setTimeoutCustom', (done) => {
    let count = 0;
    let startTime = performance.now();
    let totalTime = 0;
    let originUseSettimeoutcustom: boolean;

    originUseSettimeoutcustom = env().getBool('USE_SETTIMEOUTCUSTOM');
    env().set('USE_SETTIMEOUTCUSTOM', true);
    env().platform.setTimeoutCustom(_testSetTimeoutCustom, 0);

    function _testSetTimeoutCustom() {
      const endTime = performance.now();
      count++;
      if (count > skipCount) {
        totalTime += endTime - startTime;
      }
      if (count === totalCount) {
        const averageTime = totalTime / (totalCount - skipCount);
        console.log(`averageTime of setTimeoutCustom is ${averageTime} ms`);
        expect(averageTime).toBeLessThan(4);
        done();
        env().set('USE_SETTIMEOUTCUSTOM', originUseSettimeoutcustom);
        return;
      }
      startTime = performance.now();
      env().platform.setTimeoutCustom(_testSetTimeoutCustom, 0);
    }
  });

  it('isTypedArray returns false if not a typed array', () => {
    const platform = new PlatformBrowser();
    expect(platform.isTypedArray([1, 2, 3])).toBeFalse();
  });

  for (const typedArrayConstructor of [Float32Array, Int32Array, Uint8Array,
      Uint8ClampedArray]) {
    it(`isTypedArray returns true if it is a ${typedArrayConstructor.name}`,
       () => {
         const platform = new PlatformBrowser();
         const array = new typedArrayConstructor([1,2,3]);
         expect(platform.isTypedArray(array)).toBeTrue();
       });
  }
});
