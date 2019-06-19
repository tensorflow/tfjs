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

import {ENV} from '../environment';
import {describeWithFlags, NODE_ENVS} from '../jasmine_util';

import * as platform_node from './platform_node';
import {PlatformNode} from './platform_node';

describeWithFlags('PlatformNode', NODE_ENVS, () => {
  it('fetch should use global.fetch if defined', async () => {
    const globalFetch = ENV.global.fetch;

    spyOn(ENV.global, 'fetch').and.returnValue(() => {});

    const platform = new PlatformNode();

    await platform.fetch('test/url', {method: 'GET'});

    expect(ENV.global.fetch).toHaveBeenCalledWith('test/url', {method: 'GET'});

    ENV.global.fetch = globalFetch;
  });

  it('fetch should use node-fetch with ENV.global.fetch is null', async () => {
    const globalFetch = ENV.global.fetch;
    ENV.global.fetch = null;

    const platform = new PlatformNode();

    const savedFetch = platform_node.systemFetch;

    // Null out the system fetch so we force it to require node-fetch.
    // @ts-ignore
    platform_node.systemFetch = null;

    const testFetch = {fetch: (url: string, init: RequestInit) => {}};

    // Mock the actual fetch call.
    spyOn(testFetch, 'fetch').and.returnValue(() => {});
    // Mock the import to override the real require of node-fetch.
    spyOn(platform_node.getNodeFetch, 'importFetch')
        .and.callFake(
            () => (url: string, init: RequestInit) =>
                testFetch.fetch(url, init));

    await platform.fetch('test/url', {method: 'GET'});

    expect(platform_node.getNodeFetch.importFetch).toHaveBeenCalled();
    expect(testFetch.fetch).toHaveBeenCalledWith('test/url', {method: 'GET'});

    // @ts-ignore
    platform_node.systemFetch = savedFetch;
    ENV.global.fetch = globalFetch;
  });

  it('encodeUTF8 single string', () => {
    const platform = new PlatformNode();
    const bytes = platform.encodeUTF8('hello');
    expect(bytes.length).toBe(5);
    expect(bytes).toEqual(new Uint8Array([104, 101, 108, 108, 111]));
  });

  it('encodeUTF8 two strings delimited', () => {
    const platform = new PlatformNode();
    const bytes = platform.encodeUTF8('hello\x00world');
    expect(bytes.length).toBe(11);
    expect(bytes).toEqual(
        new Uint8Array([104, 101, 108, 108, 111, 0, 119, 111, 114, 108, 100]));
  });

  it('encodeUTF8 cyrillic', () => {
    const platform = new PlatformNode();
    const bytes = platform.encodeUTF8('Здраво');
    expect(bytes.length).toBe(12);
    expect(bytes).toEqual(new Uint8Array(
        [208, 151, 208, 180, 209, 128, 208, 176, 208, 178, 208, 190]));
  });

  it('decodeUTF8 single string', () => {
    const platform = new PlatformNode();
    const s = platform.decodeUTF8(new Uint8Array([104, 101, 108, 108, 111]));
    expect(s.length).toBe(5);
    expect(s).toEqual('hello');
  });

  it('decodeUTF8 two strings delimited', () => {
    const platform = new PlatformNode();
    const s = platform.decodeUTF8(
        new Uint8Array([104, 101, 108, 108, 111, 0, 119, 111, 114, 108, 100]));
    expect(s.length).toBe(11);
    expect(s).toEqual('hello\x00world');
  });

  it('decodeUTF8 cyrillic', () => {
    const platform = new PlatformNode();
    const s = platform.decodeUTF8(new Uint8Array(
        [208, 151, 208, 180, 209, 128, 208, 176, 208, 178, 208, 190]));
    expect(s.length).toBe(6);
    expect(s).toEqual('Здраво');
  });
});
