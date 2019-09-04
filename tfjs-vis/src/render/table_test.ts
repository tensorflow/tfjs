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

import {table} from './table';

function getRowHTML(row: Element) {
  return Array.from(row.querySelectorAll('td')).map(r => r.innerHTML);
}

function getRowText(row: Element) {
  return Array.from(row.querySelectorAll('td')).map(r => r.textContent);
}

describe('renderTable', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
  });

  it('renders a table', () => {
    const headers = [
      'Col1',
      'Col 2',
      '<em>Column 3</em>',
    ];

    const values = [
      [1, 2, 3],
      ['4', '5', '6'],
      ['<strong>7</strong>', true, false],
    ];

    const container = document.getElementById('container');
    table(container, {headers, values});

    expect(document.querySelectorAll('.tf-table').length).toBe(1);
    expect(document.querySelectorAll('.tf-table thead tr').length).toBe(1);

    const headerEl = document.querySelectorAll('.tf-table thead tr th');
    expect(headerEl[0].innerHTML).toEqual('Col1');
    expect(headerEl[1].innerHTML).toEqual('Col 2');
    expect(headerEl[2].innerHTML).toEqual('<em>Column 3</em>');
    expect(headerEl[2].textContent).toEqual('Column 3');

    expect(document.querySelectorAll('.tf-table tbody tr').length).toBe(3);

    const rows = document.querySelectorAll('.tf-table tbody tr');
    expect(getRowHTML(rows[0])).toEqual(['1', '2', '3']);
    expect(getRowHTML(rows[1])).toEqual(['4', '5', '6']);
    expect(getRowHTML(rows[2])).toEqual([
      '<strong>7</strong>', 'true', 'false'
    ]);
    expect(getRowText(rows[2])).toEqual(['7', 'true', 'false']);
  });

  it('requires necessary param', () => {
    const container = document.getElementById('container');

    // @ts-ignore
    expect(() => table({headers: []}, container)).toThrow();
    // @ts-ignore
    expect(() => table({values: [[]]}, container)).toThrow();
    // @ts-ignore
    expect(() => table({}, container)).toThrow();
  });

  it('should not throw on empty table', () => {
    const container = document.getElementById('container');
    const headers: string[] = [];
    const values: string[][] = [];

    expect(() => table(container, {headers, values})).not.toThrow();

    expect(document.querySelectorAll('.tf-table').length).toBe(1);
    expect(document.querySelectorAll('.tf-table thead tr').length).toBe(1);
    expect(document.querySelectorAll('.tf-table tbody tr').length).toBe(0);
  });
});
