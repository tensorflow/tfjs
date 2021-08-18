import {Tensor2D} from '@tensorflow/tfjs-core';

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
export interface StyleOptions {
  width?: string;
  height?: string;
  maxWidth?: string;
  maxHeight?: string;
}

/**
 * @docalias HTMLElement|{name: string, tab?: string}|Surface|{drawArea:
 * HTMLElement}
 */
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
  /**
   * Width of chart in px
   */
  width?: number;
  /**
   * Height of chart in px
   */
  height?: number;
  /**
   * Label for xAxis
   */
  xLabel?: string;
  /**
   * Label for yAxis
   */
  yLabel?: string;
  /**
   * Fontsize in px
   */
  fontSize?: number;
  /**
   * Will be set automatically
   */
  xType?: 'quantitative'|'ordinal'|'nominal';
  /**
   * Will be set automatically
   */
  yType?: 'quantitative'|'ordinal'|'nominal';
}

/**
 * Options for XY plots
 */
export interface XYPlotOptions extends VisOptions {
  /**
   * domain of the x axis. Overriden by zoomToFit
   */
  xAxisDomain?: [number, number];
  /**
   * domain of the y axis. Overriden by zoomToFit
   */
  yAxisDomain?: [number, number];
  /**
   * Set the chart bounds to just fit the data. This may modify the axis scales
   * but allows fitting more data into view.
   */
  zoomToFit?: boolean;
  /**
   * Colors to for each series plotted. An array of valid CSS color strings.
   */
  seriesColors?: string[];
}

/**
 * Data format for XY plots
 */
export interface XYPlotData {
  /**
   * An array (or nested array) of {x, y} tuples.
   */
  values: Point2D[][]|Point2D[];
  /**
   * Series names/labels
   */
  series?: string[];
}

/**
 * Histogram options.
 */
export interface HistogramOpts extends VisOptions {
  /**
   * By default a histogram will also compute and display summary statistics.
   * If stats is set to false then summary statistics will not be displayed.
   *
   * Pre computed stats can also be passed in and should have the following
   * format:
   *  {
   *    numVals?: number,
   *    min?: number,
   *    max?: number,
   *    numNans?: number,
   *    numZeros?: number,
   *    numInfs?: number,
   *  }
   */
  stats?: HistogramStats|false;

  /**
   * Maximum number of bins in histogram.
   */
  maxBins?: number;

  /**
   * Fill color for bars. Should be a valid CSS color string
   */
  color?: string;
}

/**
 * Bar chart options.
 */
export interface BarChartOpts extends VisOptions {
  /**
   * Fill color for bars. Should be a valid CSS color string
   */
  color?: string|string[];
}

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
 * An object with a 'values' property and a 'labels' property.
 */
export interface ConfusionMatrixData {
  /**
   * a square matrix of numbers representing counts for each (label, prediction)
   * pair
   */
  values: number[][];

  /**
   * Human readable labels for each class in the matrix. Optional
   */
  tickLabels?: string[];
}

export interface ConfusionMatrixOptions extends VisOptions {
  /**
   * Color cells on the diagonal. Defaults to true
   */
  shadeDiagonal?: boolean;
  /**
   * render the values of each cell as text. Defaults to true
   */
  showTextOverlay?: boolean;
  /**
   * Output range of color scale. Either a 2 element array of valid
   * css color or one of 'greyscale'|'viridis'|'blues'
   */
  colorMap?: NamedColorMap|[string, string];
}

/**
 * Datum format for scatter and line plots
 */
export interface Point2D {
  x: number;
  y: number;
}

/**
 *  An object with a 'values' property and a 'labels' property.
 */
export interface HeatmapData {
  /**
   * Matrix of values in column-major order.
   *
   * Row major order is supported by setting a boolean in options.
   */
  values: number[][]|Tensor2D;
  /**
   * x axis tick labels
   */
  xTickLabels?: string[];
  /**
   * y axis tick labels
   */
  yTickLabels?: string[];
}

/**
 * Color map names.
 */
/** @docinline */
export type NamedColorMap = 'greyscale'|'viridis'|'blues';

/**
 * Visualization options for Heatmap
 */
export interface HeatmapOptions extends VisOptions {
  /**
   * Defaults to viridis
   */
  colorMap?: NamedColorMap;

  /**
   * Custom input domain for the color scale.
   * Useful if you want to plot multiple heatmaps using the same scale.
   */
  domain?: number[];

  /**
   * Pass in data values in row-major order.
   *
   * Internally this will transpose the data values before rendering.
   */
  rowMajor?: boolean;
}

/**
 * Data format for render.table
 */
export interface TableData {
  /**
   * Column names
   */
  headers: string[];

  /**
   * An array of arrays (one for  each row). The inner
   * array length usually matches the length of data.headers.
   *
   * Typically the values are numbers or strings.
   */
  // tslint:disable-next-line:no-any
  values: any[][];
}
