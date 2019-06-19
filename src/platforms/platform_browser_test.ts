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

import {BROWSER_ENVS, describeWithFlags} from '../jasmine_util';

import {PlatformBrowser} from './platform_browser';

describeWithFlags('PlatformBrowser', BROWSER_ENVS, async () => {
  it('fetch calls window.fetch', async () => {
    const response = new Response();
    spyOn(self, 'fetch').and.returnValue(response);
    const platform = new PlatformBrowser();

    await platform.fetch('test/url', {method: 'GET'});

    expect(self.fetch).toHaveBeenCalledWith('test/url', {method: 'GET'});
  });

  it('encodeUTF8 single string', () => {
    const platform = new PlatformBrowser();
    const bytes = platform.encodeUTF8('hello');
    expect(bytes.length).toBe(5);
    expect(bytes).toEqual(new Uint8Array([104, 101, 108, 108, 111]));
  });

  it('encodeUTF8 two strings delimited', () => {
    const platform = new PlatformBrowser();
    const bytes = platform.encodeUTF8('hello\x00world');
    expect(bytes.length).toBe(11);
    expect(bytes).toEqual(
        new Uint8Array([104, 101, 108, 108, 111, 0, 119, 111, 114, 108, 100]));
  });

  it('encodeUTF8 cyrillic', () => {
    const platform = new PlatformBrowser();
    const bytes = platform.encodeUTF8('Здраво');
    expect(bytes.length).toBe(12);
    expect(bytes).toEqual(new Uint8Array(
        [208, 151, 208, 180, 209, 128, 208, 176, 208, 178, 208, 190]));
  });

  it('decodeUTF8 single string', () => {
    const platform = new PlatformBrowser();
    const s = platform.decodeUTF8(new Uint8Array([104, 101, 108, 108, 111]));
    expect(s.length).toBe(5);
    expect(s).toEqual('hello');
  });

  it('decodeUTF8 two strings delimited', () => {
    const platform = new PlatformBrowser();
    const s = platform.decodeUTF8(
        new Uint8Array([104, 101, 108, 108, 111, 0, 119, 111, 114, 108, 100]));
    expect(s.length).toBe(11);
    expect(s).toEqual('hello\x00world');
  });

  it('decodeUTF8 cyrillic', () => {
    const platform = new PlatformBrowser();
    const s = platform.decodeUTF8(new Uint8Array(
        [208, 151, 208, 180, 209, 128, 208, 176, 208, 178, 208, 190]));
    expect(s.length).toBe(6);
    expect(s).toEqual('Здраво');
  });
});
