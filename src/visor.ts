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

import {VisorComponent} from './components/visor';
import {SurfaceInfo, SurfaceInfoStrict, Visor} from './types';

let visorSingleton: Visor;
const DEFAULT_TAB = 'Visor';
const VISOR_CONTAINER_ID = 'tfjs-visor-container';

/**
 * The primary interface to the visor is the visor() function.
 *
 * This returns a singleton object with the public API of the visor. The
 * singleton object will be replaced if the visor is removed from the DOM for
 * some reason.
 */
export function visor(): Visor {
  if (typeof document === 'undefined') {
    throw new Error(
        'No document defined. This library needs a browser/dom to work');
  }

  if (document.getElementById(VISOR_CONTAINER_ID) && visorSingleton != null) {
    return visorSingleton;
  }

  // Create the container
  let visorEl = document.getElementById(VISOR_CONTAINER_ID);

  if (visorEl == null) {
    visorEl = document.createElement('div');
    visorEl.id = VISOR_CONTAINER_ID;
    document.body.appendChild(visorEl);
  }

  let renderRoot: Element;
  function renderVisor(
      domNode: HTMLElement,
      surfaceList: Map<string, SurfaceInfoStrict>): VisorComponent {
    let visorInstance: VisorComponent;
    renderRoot = VisorComponent.render(domNode, renderRoot, {
      ref: (r: VisorComponent) => visorInstance = r,
      surfaceList: Array.from(surfaceList.values()),
    });
    // Side effect of VisorComponent.render() is to assign visorInstance
    return visorInstance!;
  }

  // TODO: consider changing this type. Possibly lift into top level state
  // object
  const surfaceList: Map<string, SurfaceInfoStrict> = new Map();
  const visorComponentInstance: VisorComponent =
      renderVisor(visorEl, surfaceList);

  // Singleton visor instance. Implements public API of the visor.
  visorSingleton = {
    el: visorEl,
    surface: (options: SurfaceInfo) => {
      const {name} = options;
      const tab = options.tab == null ? DEFAULT_TAB : options.tab;

      if (name == null ||
          // tslint:disable-next-line
          !(typeof name === 'string' || name as any instanceof String)) {
        throw new Error(
            // tslint:disable-next-line
            'You must pass a config object with a \'name\' property to create or retrieve a surface');
      }

      const finalOptions: SurfaceInfoStrict = {
        ...options,
        tab,
      };

      const key = `${name}-${tab}`;
      if (!surfaceList.has(key)) {
        surfaceList.set(key, finalOptions);
      }

      renderVisor(visorEl as HTMLElement, surfaceList);
      return visorComponentInstance.getSurface(name, tab);
    },
    isFullscreen: () => visorComponentInstance.isFullscreen(),
    isOpen: () => visorComponentInstance.isOpen(),
    close: () => visorComponentInstance.close(),
    open: () => visorComponentInstance.open(),
    toggle: () => visorComponentInstance.toggle(),
    toggleFullScreen: () => visorComponentInstance.toggleFullScreen(),
    bindKeys: () => {
      throw new Error('Not yet implemented');
    },
    unbindKeys: () => {
      throw new Error('Not yet implemented');
    },
  };

  return visorSingleton;
}
