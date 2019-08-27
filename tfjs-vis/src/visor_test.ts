/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {visor} from './index';

const tick = (ms = 1) => new Promise(resolve => setTimeout(resolve, ms));

describe('Visor Singleton', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('renders an empty visor', () => {
    visor();
    expect(document.querySelectorAll('.visor').length).toBe(1);
  });

  it('visor.el is an HTMLElement', () => {
    const visorInstance = visor();
    expect(visorInstance.el instanceof HTMLElement).toBe(true);
  });

  it('renders only one visor', () => {
    const v1 = visor();
    const v2 = visor();
    const v3 = visor();
    expect(document.querySelectorAll('.visor').length).toBe(1);
    expect(v1).toEqual(v2);
    expect(v1).toEqual(v3);
  });

  it('adds a surface', () => {
    const visorInstance = visor();
    visorInstance.surface({name: 'surface 1', tab: 'tab 1'});
    expect(document.querySelectorAll('.tf-surface').length).toBe(1);
    expect(document.querySelector('.tf-surface').textContent)
        .toEqual('surface 1');

    expect(document.querySelectorAll('.tf-tab').length).toBe(1);
    expect(document.querySelector('.tf-tab').textContent).toEqual('tab 1');
  });

  it('requires a surface name', () => {
    const visorInstance = visor();
    expect(() => {
      // @ts-ignore
      visorInstance.surface();
    }).toThrow();

    expect(() => {
      // @ts-ignore
      visorInstance.surface('Incorrect Name Param');
    }).toThrow();

    expect(() => {
      // @ts-ignore
      visorInstance.surface({notName: 'Incorrect Name Param'});
    }).toThrow();
  });

  it('retrieves a surface', () => {
    const visorInstance = visor();
    const s1 = visorInstance.surface({name: 'surface 1', tab: 'tab 1'});
    expect(document.querySelectorAll('.tf-surface').length).toBe(1);
    expect(document.querySelector('.tf-surface').textContent)
        .toEqual('surface 1');

    const s2 = visorInstance.surface({name: 'surface 1', tab: 'tab 1'});
    expect(document.querySelectorAll('.tf-surface').length).toBe(1);
    expect(document.querySelector('.tf-surface').textContent)
        .toEqual('surface 1');

    expect(s1).toEqual(s2);
  });

  it('adds a surface with the default tab', () => {
    const visorInstance = visor();
    visorInstance.surface({name: 'surface1'});

    expect(document.querySelectorAll('.tf-tab').length).toBe(1);
    expect(document.querySelector('.tf-tab').textContent).toEqual('Visor');
  });

  it('adds two surfaces', () => {
    const visorInstance = visor();
    const s1 = visorInstance.surface({name: 'surface 1', tab: 'tab 1'});
    const s2 = visorInstance.surface({name: 'surface 2', tab: 'tab 1'});

    expect(s1).not.toEqual(s2);

    const surfaces = document.querySelectorAll('.tf-surface');
    expect(surfaces.length).toBe(2);
    expect(document.querySelectorAll('.tf-tab').length).toBe(1);

    expect(surfaces[0].textContent).toEqual('surface 1');
    expect(surfaces[1].textContent).toEqual('surface 2');
  });

  it('switches tabs on surface addition', () => {
    let tabs;
    const visorInstance = visor();

    visorInstance.surface({name: 'surface 1', tab: 'tab 1'});
    tabs = document.querySelectorAll('.tf-tab');
    expect(tabs[0].getAttribute('data-isactive')).toEqual('true');

    visorInstance.surface({name: 'surface 2', tab: 'tab 2'});
    tabs = document.querySelectorAll('.tf-tab');
    expect(tabs[1].getAttribute('data-isactive')).toEqual('true');
    expect(tabs[0].getAttribute('data-isactive')).toBeFalsy();

    visorInstance.surface({name: 'surface 3', tab: 'tab 3'});
    tabs = document.querySelectorAll('.tf-tab');
    expect(tabs[2].getAttribute('data-isactive')).toEqual('true');
    expect(tabs[0].getAttribute('data-isactive')).toBeFalsy();
    expect(tabs[1].getAttribute('data-isactive')).toBeFalsy();
  });

  it('closes/opens', async () => {
    const visorInstance = visor();

    expect(document.querySelector('.visor').getAttribute('data-isopen'))
        .toBe('true');
    expect(visorInstance.isOpen()).toBe(true);

    visorInstance.close();
    await tick();
    expect(document.querySelector('.visor').getAttribute('data-isopen'))
        .toBeFalsy();
    expect(visorInstance.isOpen()).toBe(false);

    visorInstance.open();
    await tick();
    expect(document.querySelector('.visor').getAttribute('data-isopen'))
        .toBe('true');
    expect(visorInstance.isOpen()).toBe(true);
  });

  it('toggles', async () => {
    const visorInstance = visor();

    expect(document.querySelector('.visor').getAttribute('data-isopen'))
        .toBe('true');
    expect(visorInstance.isOpen()).toBe(true);

    visorInstance.toggle();
    await tick();
    expect(document.querySelector('.visor').getAttribute('data-isopen'))
        .toBeFalsy();
    expect(visorInstance.isOpen()).toBe(false);

    visorInstance.toggle();
    await tick();
    expect(document.querySelector('.visor').getAttribute('data-isopen'))
        .toBe('true');
    expect(visorInstance.isOpen()).toBe(true);
  });

  it('fullscreen toggles', async () => {
    const visorInstance = visor();
    expect(visorInstance.isOpen()).toBe(true);

    expect(document.querySelector('.visor').getAttribute('data-isfullscreen'))
        .toBeFalsy();

    visorInstance.toggleFullScreen();
    await tick();
    expect(document.querySelector('.visor').getAttribute('data-isfullscreen'))
        .toBe('true');

    visorInstance.toggleFullScreen();
    await tick();
    expect(document.querySelector('.visor').getAttribute('data-isfullscreen'))
        .toBeFalsy();
  });

  it('sets the active tab', async () => {
    let tabs;
    const visorInstance = visor();

    visorInstance.surface({name: 'surface 1', tab: 'tab 1'});
    visorInstance.surface({name: 'surface 2', tab: 'tab 2'});
    visorInstance.surface({name: 'surface 2', tab: 'tab 3'});

    tabs = document.querySelectorAll('.tf-tab');
    expect(tabs[2].getAttribute('data-isactive')).toEqual('true');

    visorInstance.setActiveTab('tab 2');
    await tick();
    tabs = document.querySelectorAll('.tf-tab');
    expect(tabs[1].getAttribute('data-isactive')).toEqual('true');

    visorInstance.setActiveTab('tab 1');
    await tick();
    tabs = document.querySelectorAll('.tf-tab');
    expect(tabs[0].getAttribute('data-isactive')).toEqual('true');

    visorInstance.setActiveTab('tab 3');
    await tick();
    tabs = document.querySelectorAll('.tf-tab');
    expect(tabs[2].getAttribute('data-isactive')).toEqual('true');
  });

  it('throws error if tab does not exist', () => {
    const visorInstance = visor();

    visorInstance.surface({name: 'surface 1', tab: 'tab 1'});
    visorInstance.surface({name: 'surface 2', tab: 'tab 2'});

    expect(() => {
      visorInstance.setActiveTab('not present');
    }).toThrow();
  });

  it('unbinds keyboard handler', () => {
    const visorInstance = visor();

    const BACKTICK_KEY = 192;
    const event = document.createEvent('Event');
    event.initEvent('keydown', true, true);
    // @ts-ignore
    event['keyCode'] = BACKTICK_KEY;

    document.dispatchEvent(event);
    expect(visorInstance.isOpen()).toBe(false);
    document.dispatchEvent(event);
    expect(visorInstance.isOpen()).toBe(true);

    // Unbind keys
    visorInstance.unbindKeys();
    document.dispatchEvent(event);
    expect(visorInstance.isOpen()).toBe(true);
    document.dispatchEvent(event);
    expect(visorInstance.isOpen()).toBe(true);
  });

  it('rebinds keyboard handler', () => {
    const visorInstance = visor();

    const BACKTICK_KEY = 192;
    const event = document.createEvent('Event');
    event.initEvent('keydown', true, true);
    // @ts-ignore
    event['keyCode'] = BACKTICK_KEY;

    // Unbind keys
    visorInstance.unbindKeys();
    document.dispatchEvent(event);
    expect(visorInstance.isOpen()).toBe(true);
    document.dispatchEvent(event);
    expect(visorInstance.isOpen()).toBe(true);

    // rebind keys
    visorInstance.bindKeys();
    document.dispatchEvent(event);
    expect(visorInstance.isOpen()).toBe(false);
    document.dispatchEvent(event);
    expect(visorInstance.isOpen()).toBe(true);
  });
});
