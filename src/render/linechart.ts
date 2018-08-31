import embed, {Mode, VisualizationSpec} from 'vega-embed';

import {Drawable, Point2D, VisOptions} from '../types';

import {getDrawArea} from './render_utils';

/**
 * Renders a line chart
 *
 * @param data Data in the following format
 *  {
 *    // A nested array of objects each with an x and y property,
 *    // one per series.
 *    // If you only have one series to render you can just pass an array
 *    // of objects with x, y properties
 *    values: {x: number, y: number}[][]
 *
 *    // An array of strings with the names of each series passed above.
 *    // Optional
 *    series: string[]
 *  }
 * @param container An HTMLElement in which to draw the chart
 * @param opts optional parameters
 * @param opts.width width of chart in px
 * @param opts.height height of chart in px
 * @param opts.xLabel label for x axis
 * @param opts.yLabel label for y axis
 */
export async function renderLinechart(
    data: {values: Point2D[][]|Point2D[], series?: string[]},
    container: Drawable, opts: VisOptions = {}): Promise<void> {
  let inputArray = data.values;
  const _series = data.series == null ? [] : data.series;

  // Nest data if necessary before further processing
  inputArray = Array.isArray(inputArray[0]) ? inputArray as Point2D[][] :
                                              [inputArray] as Point2D[][];

  const values = inputArray.reduce((memo, seriesData, i) => {
    const seriesName: string =
        _series[i] != null ? _series[i] : `Series ${i + 1}`;
    const seriesVals =
        seriesData.map(v => Object.assign({}, v, {series: seriesName}));
    return memo.concat(seriesVals);
  }, []);

  const drawArea = getDrawArea(container);
  const options = Object.assign({}, defaultOpts, opts);

  const embedOpts = {
    actions: false,
    mode: 'vega-lite' as Mode,
  };

  const encodings: {} = {
    'x': {
      'field': 'x',
      'type': options.xType,
      'title': options.xLabel,
    },
    'y': {
      'field': 'y',
      'type': options.yType,
      'title': options.yLabel,
    },
    'color': {
      'field': 'series',
      'type': 'nominal',
    },
  };

  const spec: VisualizationSpec = {
    'width': options.width || drawArea.clientWidth,
    'height': options.height || drawArea.clientHeight,
    'padding': 5,
    'autosize': {
      'type': 'fit',
      'contains': 'padding',
      'resize': true,
    },
    'data': {'values': values},
    'layer': [
      {
        // Render the main line chart
        'mark': {'type': 'line'},
        'encoding': encodings,
      },
      {
        'selection': {
          'nearestPoint': {
            'type': 'single',
            'on': 'mouseover',
            'nearest': true,
            'empty': 'none',
            'encodings': ['x']
          },
        },
        // Draw a point where the selection is
        'mark': {'type': 'point'},
        'encoding': Object.assign({}, encodings, {
          'opacity': {
            'value': 0,
            'condition': {
              'selection': 'nearestPoint',
              'value': 1,
            },
          }
        }),
      },
      {
        // Draw a vertical line where the selection is
        'transform': [{'filter': {'selection': 'nearestPoint'}}],
        'mark': {'type': 'rule', 'color': 'gray'},
        'encoding': {
          'x': {
            'type': options.xType,
            'field': 'x',
          }
        }
      },
      {
        // Render a tooltip where the selection is
        'transform': [{'filter': {'selection': 'nearestPoint'}}],
        'mark': {
          'type': 'text',
          'align': 'left',
          'dx': 5,
          'dy': -5,
          'color': 'black',
        },
        'encoding': Object.assign({}, encodings, {
          'text': {
            'type': options.xType,
            'field': 'y',
            'format': '.6f',
          },
          // Unset text color to improve readability
          'color': undefined,
        }),
      },

    ],
  };

  await embed(drawArea, spec, embedOpts);
  return Promise.resolve();
}

const defaultOpts = {
  xLabel: 'x',
  yLabel: 'y',
  xType: 'quantitative',
  yType: 'quantitative',
};
