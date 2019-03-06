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

import {format as d3Format} from 'd3-format';
import {select as d3Select} from 'd3-selection';
import {css} from 'glamor';
import {tachyons as tac} from 'glamor-tachyons';
import {Drawable} from '../types';
import {getDrawArea} from './render_utils';

/**
 * Renders a table
 *
 * ```js
 * const headers = [
 *  'Col 1',
 *  'Col 2',
 *  'Col 3',
 * ];
 *
 * const values = [
 *  [1, 2, 3],
 *  ['4', '5', '6'],
 *  ['strong>7</strong>', true, false],
 * ];
 *
 * const surface = { name: 'Table', tab: 'Charts' };
 * tfvis.render.table(surface, { headers, values });
 * ```
 *
 * @param data Data in the following format
 *    {
 *      headers: string[],
 *      values:  any[][],
 *    }
 *    data.headers are the column names
 *    data.values is an array of arrays (one for  each row). The inner
 *    array length usually matches the length of data.headers. Usually
 *    the values are strings or numbers, these are inserted as html
 *    content so html strings are also supported.
 *
 * @param container An `HTMLElement` or `Surface` in which to draw the table.
 *    Note that the chart expects to have complete control over
 *    the contents of the container and can clear its contents
 *    at will.
 * @param opts.fontSize fontSize in pixels for text in the chart.
 *
 */
/** @doc {heading: 'Charts', namespace: 'render'} */
export function table(
    container: Drawable,
    // tslint:disable-next-line:no-any
    data: {headers: string[], values: any[][]},
    opts: {fontSize?: number} = {}) {
  if (data && data.headers == null) {
    throw new Error('Data to render must have a "headers" property');
  }

  if (data && data.values == null) {
    throw new Error('Data to render must have a "values" property');
  }

  const drawArea = getDrawArea(container);

  const options = Object.assign({}, defaultOpts, opts);

  let table = d3Select(drawArea).select('table.tf-table');

  const tableStyle = css({
    ...tac('f6 w-100 mw8 center'),
    fontSize: options.fontSize,
  });

  // If a table is not already present on this element add one
  if (table.size() === 0) {
    table = d3Select(drawArea).append('table');

    table.attr('class', ` ${tableStyle} tf-table`);

    table.append('thead').append('tr');
    table.append('tbody');
  }

  if (table.size() !== 1) {
    throw new Error('Error inserting table');
  }

  //
  // Add the reader row
  //
  const headerRowStyle =
      css({...tac('fw6 bb b--black-20 tl pb3 pr3 bg-white')});
  const headers =
      table.select('thead').select('tr').selectAll('th').data(data.headers);
  const headersEnter =
      headers.enter().append('th').attr('class', `${headerRowStyle}`);
  headers.merge(headersEnter).html(d => d);

  headers.exit().remove();

  //
  // Add the data rows
  //
  const format = d3Format(',.4~f');

  const rows = table.select('tbody').selectAll('tr').data(data.values);
  const rowsEnter = rows.enter().append('tr');

  // Nested selection to add individual cells
  const cellStyle = css({...tac('pa1 bb b--black-20')});
  const cells = rows.merge(rowsEnter).selectAll('td').data(d => d);
  const cellsEnter = cells.enter().append('td').attr('class', `${cellStyle}`);
  cells.merge(cellsEnter).html(d => typeof d === 'number' ? format(d) : d);

  cells.exit().remove();
  rows.exit().remove();
}

const defaultOpts = {
  fontSize: 14,
};
