/*
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

// Types shared across the project and that users will commonly interact with

/**
 * Visor public API
 */
export interface Visor {
  /**
   * The containing HTMLElement
   */
  el: HTMLElement;

  /**
   * Returns a surface, creating one if necessary
   */
  surface: (options: SurfaceInfo) => Surface;

  /**
   * Returns true if the visor is in fullscreen mode. Note that the visor
   * may be in a closed state even if it is in fullscreen mode
   */
  isFullscreen: () => boolean;

  /**
   * Returns true if the visor is currently open/visible false otherwise
   */
  isOpen: () => boolean;

  /**
   * Opens the visor
   */
  open: () => void;

  /**
   * Closes the visor
   */
  close: () => void;

  /**
   * toggles the visor open and closed
   */
  toggle: () => void;

  /**
   * toggles the fullscreen mode of the visor
   */
  toggleFullScreen: () => void;

  /**
   * Unbinds the default keyboard shortcuts
   */
  unbindKeys: () => void;

  /**
   * Binds the default keyboard shortcuts
   */
  bindKeys: () => void;
}

/**
 * The public api of a 'surface'
 */
export interface Surface {
  /**
   * The containing HTML element for this surface
   */
  container: HTMLElement;

  /**
   * A textual label for the surface.
   */
  label: HTMLElement;

  /**
   * A container for plots and other renderings
   */
  drawArea: HTMLElement;
}

/**
 * Options used to specify a surface.
 *
 * name and tab are also used for retrieval of a surface instance.
 */
export interface SurfaceInfo {
  /**
   * The name / label of this surface
   */
  name: string;

  /**
   * The name of the tab this surface should appear on
   */
  tab?: string;

  /**
   * Display Styles for the surface
   */
  styles?: StyleOptions;
}

/**
 * Internally all surfaces must have a tab.
 */
export interface SurfaceInfoStrict extends SurfaceInfo {
  name: string;
  tab: string;
  styles?: StyleOptions;
}

/**
 * Style properties are generally optional as components will specify defaults.
 */
export type StyleOptions = Partial<CSSOptions>;
export interface CSSOptions {
  width: string;
  height: string;
  maxWidth: string;
  maxHeight: string;
}

export type Drawable = HTMLElement|Surface|{
  drawArea: HTMLElement;
};

/**
 * Common visualisation options for '.render' functions.
 */
export interface VisOptions {
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xType?: 'quantitative'|'ordinal'|'nominal';
  yType?: 'quantitative'|'ordinal'|'nominal';
}

/**
 * Histogram options.
 */
export type HistogramOpts = VisOptions&{
  stats?: HistogramStats|false;
  maxBins?: number;
};

/**
 * Summary statistics for histogram.
 */
export interface HistogramStats {
  numVals?: number;
  min?: number;
  max?: number;
  numNans?: number;
  numZeros?: number;
}

export type TypedArray = Int8Array|Uint8Array|Int16Array|Uint16Array|Int32Array|
    Uint32Array|Uint8ClampedArray|Float32Array|Float64Array;

export interface ConfusionMatrixData {
  values: number[][];
  labels?: string[];
}

/**
 * Datum format for scatter and line plots
 */
export type XYVal = {
  x: number; y: number;
};
