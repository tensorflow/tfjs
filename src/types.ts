import {Tensor2D} from '@tensorflow/tfjs';

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

export type Drawable = HTMLElement|Surface|SurfaceInfo|{
  drawArea: HTMLElement;
};

export function isSurfaceInfo(drawable: Drawable): drawable is SurfaceInfo {
  if ((drawable as SurfaceInfo).name != null) {
    return true;
  }
  return false;
}

export function isSurface(drawable: Drawable): drawable is Surface {
  if ((drawable as Surface).drawArea instanceof HTMLElement) {
    return true;
  }
  return false;
}

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
  fontSize?: number;
}

/**
 * Options for XY plots
 */
export interface XYPlotOptions extends VisOptions {
  xAxisDomain?: [number, number];
  yAxisDomain?: [number, number];
  zoomToFit?: boolean;
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
  numInfs?: number;
}

/**
 * Type alias for typed arrays
 */
export type TypedArray = Int8Array|Uint8Array|Int16Array|Uint16Array|Int32Array|
    Uint32Array|Uint8ClampedArray|Float32Array|Float64Array;

/**
 * Data format for confusion matrix
 */
export interface ConfusionMatrixData {
  values: number[][];
  tickLabels?: string[];
}

/**
 * Datum format for scatter and line plots
 */
export type Point2D = {
  x: number; y: number;
};

/**
 * Data format for confusion matrix
 */
export interface HeatmapData {
  values: number[][]|Tensor2D;
  xTickLabels?: string[];
  yTickLabels?: string[];
}

/**
 * Color map names.
 *
 * Currently supported by heatmap
 */
export type NamedColorMap = 'greyscale'|'viridis'|'blues';

/**
 * Visualization options for Heatmap
 */
export interface HeatmapOptions extends VisOptions {
  colorMap?: NamedColorMap;
  domain?: number[];
}
