/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import * as environment from './environment';
import {Environment} from './environment';

describe('initializes flags from the url', () => {
  // Silence console.warns for these tests.
  beforeAll(() => spyOn(console, 'warn').and.returnValue(null));

  it('no overrides one registered flag', () => {
    spyOn(environment, 'getQueryParams').and.returnValue({});

    const global = {location: {search: ''}};
    const env = new Environment(global);
    env.registerFlag('FLAG1', () => false);
    expect(env.get('FLAG1')).toBe(false);
  });

  it('one unregistered flag', () => {
    spyOn(environment, 'getQueryParams').and.returnValue({
      'tfjsflags': 'FLAG1:true'
    });

    const global = {location: {search: ''}};
    const env = new Environment(global);
    expect(env.features).toEqual({});
  });

  it('one registered flag true', () => {
    const global = {location: {search: '?tfjsflags=FLAG1:true'}};
    const env = new Environment(global);
    env.registerFlag('FLAG1', () => false);

    expect(env.get('FLAG1')).toBe(true);
  });

  it('one registered flag false', () => {
    const global = {location: {search: '?tfjsflags=FLAG1:false'}};
    const env = new Environment(global);
    env.registerFlag('FLAG1', () => true);

    expect(env.get('FLAG1')).toBe(false);
  });

  it('two registered flags', () => {
    const global = {location: {search: '?tfjsflags=FLAG1:true,FLAG2:200'}};
    const env = new Environment(global);
    env.registerFlag('FLAG1', () => false);
    env.registerFlag('FLAG2', () => 100);

    expect(env.get('FLAG1')).toBe(true);
    expect(env.get('FLAG2')).toBe(200);
  });
});

describe('flag registration and evaluation', () => {
  it('one flag registered', () => {
    const env = new Environment({});

    const evalObject = {eval: () => true};
    const spy = spyOn(evalObject, 'eval').and.callThrough();

    env.registerFlag('FLAG1', () => evalObject.eval());

    expect(env.get('FLAG1')).toBe(true);
    expect(spy.calls.count()).toBe(1);

    // Multiple calls to get do not call the evaluation function again.
    expect(env.get('FLAG1')).toBe(true);
    expect(spy.calls.count()).toBe(1);
  });

  it('multiple flags registered', () => {
    const env = new Environment({});

    const evalObject = {eval1: () => true, eval2: () => 100};
    const spy1 = spyOn(evalObject, 'eval1').and.callThrough();
    const spy2 = spyOn(evalObject, 'eval2').and.callThrough();

    env.registerFlag('FLAG1', () => evalObject.eval1());
    env.registerFlag('FLAG2', () => evalObject.eval2());

    expect(env.get('FLAG1')).toBe(true);
    expect(spy1.calls.count()).toBe(1);
    expect(spy2.calls.count()).toBe(0);
    expect(env.get('FLAG2')).toBe(100);
    expect(spy1.calls.count()).toBe(1);
    expect(spy2.calls.count()).toBe(1);

    // Multiple calls to get do not call the evaluation function again.
    expect(env.get('FLAG1')).toBe(true);
    expect(env.get('FLAG2')).toBe(100);
    expect(spy1.calls.count()).toBe(1);
    expect(spy2.calls.count()).toBe(1);
  });

  it('setting overrides value', () => {
    const env = new Environment({});

    const evalObject = {eval: () => true};
    const spy = spyOn(evalObject, 'eval').and.callThrough();

    env.registerFlag('FLAG1', () => evalObject.eval());

    expect(env.get('FLAG1')).toBe(true);
    expect(spy.calls.count()).toBe(1);

    env.set('FLAG1', false);

    expect(env.get('FLAG1')).toBe(false);
    expect(spy.calls.count()).toBe(1);
  });

  it('set hook is called', () => {
    const env = new Environment({});

    const evalObject = {eval: () => true, setHook: () => true};
    const evalSpy = spyOn(evalObject, 'eval').and.callThrough();
    const setHookSpy = spyOn(evalObject, 'setHook').and.callThrough();

    env.registerFlag(
        'FLAG1', () => evalObject.eval(), () => evalObject.setHook());

    expect(env.get('FLAG1')).toBe(true);
    expect(evalSpy.calls.count()).toBe(1);
    expect(setHookSpy.calls.count()).toBe(0);

    env.set('FLAG1', false);

    expect(env.get('FLAG1')).toBe(false);
    expect(evalSpy.calls.count()).toBe(1);
    expect(setHookSpy.calls.count()).toBe(1);
  });
});

describe('environment.getQueryParams', () => {
  it('basic', () => {
    expect(environment.getQueryParams('?a=1&b=hi&f=animal'))
        .toEqual({'a': '1', 'b': 'hi', 'f': 'animal'});
  });
});
