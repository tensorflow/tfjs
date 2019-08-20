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
import {SurfaceInfo, SurfaceInfoStrict} from './types';

let visorSingleton: Visor;
const DEFAULT_TAB = 'Visor';
const VISOR_CONTAINER_ID = 'tfjs-visor-container';

/**
 * The primary interface to the visor is the visor() function.
 *
 * This returns a singleton instance of the Visor class. The
 * singleton object will be replaced if the visor is removed from the DOM for
 * some reason.
 *
 * ```js
 * // Show the visor
 * tfvis.visor();
 * ```
 *
 */
/** @doc {heading: 'Visor & Surfaces'} */
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

  visorSingleton =
      new Visor(visorComponentInstance, visorEl, surfaceList, renderVisor);

  return visorSingleton;
}

/**
 * An instance of the visor. An instance of this class is created using the
 * `visor()` function.
 */
/** @doc {heading: 'Visor & Surfaces', subheading: 'Visor Methods'} */
export class Visor {
  private visorComponent: VisorComponent;
  private surfaceList: Map<string, SurfaceInfoStrict>;
  private renderVisor:
      (domNode: HTMLElement,
       surfaceList: Map<string, SurfaceInfoStrict>) => VisorComponent;

  /**
   * The underlying html element of the visor.
   */
  public el: HTMLElement;

  constructor(
      visorComponent: VisorComponent, visorEl: HTMLElement,
      surfaceList: Map<string, SurfaceInfoStrict>,
      renderVisor:
          (domNode: HTMLElement,
           surfaceList: Map<string, SurfaceInfoStrict>) => VisorComponent) {
    this.visorComponent = visorComponent;
    this.el = visorEl;
    this.surfaceList = surfaceList;
    this.renderVisor = renderVisor;
  }

  /**
   * Creates a surface on the visor
   *
   * Most methods in tfjs-vis that take a surface also take a SurfaceInfo
   * so you rarely need to call this method unless you want to make a custom
   * plot.
   *
   * ```js
   * // Create a surface on a tab
   * tfvis.visor().surface({name: 'My Surface', tab: 'My Tab'});
   * ```
   *
   * ```js
   * // Create a surface and specify its height
   * tfvis.visor().surface({name: 'Custom Height', tab: 'My Tab', styles: {
   *    height: 500
   * }})
   * ```
   *
   * @param options
   */
  /** @doc {heading: 'Visor & Surfaces', subheading: 'Visor Methods'} */
  surface(options: SurfaceInfo) {
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
    if (!this.surfaceList.has(key)) {
      this.surfaceList.set(key, finalOptions);
    }

    this.renderVisor(this.el as HTMLElement, this.surfaceList);
    return this.visorComponent.getSurface(name, tab);
  }

  /**
   * Returns a boolean indicating if the visor is in 'fullscreen' mode
   */
  /** @doc {heading: 'Visor & Surfaces', subheading: 'Visor Methods'} */
  isFullscreen() {
    return this.visorComponent.isFullscreen();
  }

  /**
   * Returns a boolean indicating if the visor is open
   */
  /** @doc {heading: 'Visor & Surfaces', subheading: 'Visor Methods'} */
  isOpen() {
    return this.visorComponent.isOpen();
  }

  /**
   * Closes the visor.
   */
  /** @doc {heading: 'Visor & Surfaces', subheading: 'Visor Methods'} */
  close() {
    return this.visorComponent.close();
  }

  /**
   * Opens the visor.
   */
  /** @doc {heading: 'Visor & Surfaces', subheading: 'Visor Methods'} */
  open() {
    return this.visorComponent.open();
  }

  /**
   * Toggles the visor (closed vs open).
   */
  /** @doc {heading: 'Visor & Surfaces', subheading: 'Visor Methods'} */
  toggle() {
    return this.visorComponent.toggle();
  }

  /** @doc {heading: 'Visor & Surfaces', subheading: 'Visor Methods'} */
  toggleFullScreen() {
    return this.visorComponent.toggleFullScreen();
  }

  /**
   * Binds the ~ (tilde) key to toggle the visor.
   *
   * This is called by default when the visor is initially created.
   */
  /** @doc {heading: 'Visor & Surfaces', subheading: 'Visor Methods'} */
  bindKeys() {
    return this.visorComponent.bindKeys();
  }

  /**
   * Unbinds the keyboard control to toggle the visor.
   */
  /** @doc {heading: 'Visor & Surfaces', subheading: 'Visor Methods'} */
  unbindKeys() {
    return this.visorComponent.unbindKeys();
  }

  /**
   * Sets the active tab for the visor.
   */
  /** @doc {heading: 'Visor & Surfaces', subheading: 'Visor Methods'} */
  setActiveTab(tabName: string) {
    const tabs = this.visorComponent.state.tabs;
    if (!tabs.has(tabName)) {
      throw new Error(`Tab '${tabName}' does not exist`);
    }
    this.visorComponent.setState({activeTab: tabName});
  }
}
