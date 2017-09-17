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

import '../demo-header';
import '../demo-footer';

import {PolymerElement, PolymerHTMLElement} from '../polymer-spec';
import {BenchmarkRunGroup} from './benchmark';

import {getRunGroups} from './math-benchmark-run-groups';

// tslint:disable-next-line:variable-name
export let MathBenchmarkPolymer: new () => PolymerHTMLElement = PolymerElement(
    {is: 'math-benchmark', properties: {benchmarkRunGroupNames: Array}});

function getDisplayParams(params?: {}): string {
  if (params == null) {
    return '';
  }

  const kvParams = params as {[key: string]: string};
  const out: string[] = [];
  const keys = Object.keys(kvParams);
  if (keys.length === 0) {
    return '';
  }
  for (let i = 0; i < keys.length; i++) {
    out.push(keys[i] + ': ' + kvParams[keys[i]]);
  }
  return '{' + out.join(', ') + '}';
}
export class MathBenchmark extends MathBenchmarkPolymer {
  // Polymer properties.
  private benchmarkRunGroupNames: string[];
  private stopMessages: boolean[];

  ready() {
    const groups = getRunGroups();
    // Set up the benchmarks UI.
    const benchmarkRunGroupNames: string[] = [];
    this.stopMessages = [];
    for (let i = 0; i < groups.length; i++) {
      benchmarkRunGroupNames.push(
          groups[i].name + ': ' + getDisplayParams(groups[i].params));
      this.stopMessages.push(false);
    }
    this.benchmarkRunGroupNames = benchmarkRunGroupNames;

    // In a setTimeout to let the UI update before we add event listeners.
    setTimeout(() => {
      const runButtons = this.querySelectorAll('.run-test');
      const stopButtons = this.querySelectorAll('.run-stop');
      for (let i = 0; i < runButtons.length; i++) {
        runButtons[i].addEventListener('click', () => {
          this.runBenchmarkGroup(groups, i);
        });
        stopButtons[i].addEventListener('click', () => {
          this.stopMessages[i] = true;
        });
      }
    }, 0);
  }

  private runBenchmarkGroup(
      groups: BenchmarkRunGroup[], benchmarkRunGroupIndex: number) {
    const benchmarkRunGroup = groups[benchmarkRunGroupIndex];

    const canvas = this.querySelectorAll('.run-plot')[benchmarkRunGroupIndex] as
        HTMLCanvasElement;
    const context = canvas.getContext('2d') as CanvasRenderingContext2D;

    const datasets: ChartDataSets[] = [];
    for (let i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
      const hue = Math.floor(360 * i / benchmarkRunGroup.benchmarkRuns.length);
      datasets.push({
        data: benchmarkRunGroup.benchmarkRuns[i].chartData,
        fill: false,
        label: benchmarkRunGroup.benchmarkRuns[i].name,
        borderColor: 'hsl(' + hue + ', 100%, 40%)',
        backgroundColor: 'hsl(' + hue + ', 100%, 70%)',
        pointRadius: 0,
        pointHitRadius: 5,
        borderWidth: 1,
        lineTension: 0
      });
    }

    const chart = new Chart(context, {
      type: 'line',
      data: {datasets},
      options: {
        animation: {duration: 0},
        responsive: false,
        scales: {
          xAxes: [{
            type: 'linear',
            position: 'bottom',
            ticks: {
              min: benchmarkRunGroup.min,
              max: benchmarkRunGroup.max,
              stepSize: benchmarkRunGroup.stepSize,
              callback: (label: string) => {
                return benchmarkRunGroup.stepToSizeTransformation != null ?
                    benchmarkRunGroup.stepToSizeTransformation(+label) :
                    +label;
              }
              // tslint:disable-next-line:no-any
            } as any  // Note: the typings for this are incorrect, cast as any.
          }],
          yAxes: [{
            ticks: {
              callback: (label, index, labels) => {
                return label + 'ms';
              }
            },
          }]
        },
        tooltips: {mode: 'label'},
        title: {text: benchmarkRunGroup.name}
      }
    });
    canvas.style.display = 'none';

    const runMessage =
        this.querySelectorAll('.run-message')[benchmarkRunGroupIndex] as
        HTMLElement;
    runMessage.style.display = 'block';

    const runNumbersTable =
        this.querySelectorAll('.run-numbers-table')[benchmarkRunGroupIndex] as
        HTMLElement;
    runNumbersTable.innerHTML = '';
    runNumbersTable.style.display = 'none';

    // Set up the header for the table.
    const headers = ['size'];
    for (let i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
      headers.push(benchmarkRunGroup.benchmarkRuns[i].name);
    }
    runNumbersTable.appendChild(this.buildRunNumbersRow(headers));

    this.runBenchmarkSteps(
        chart, benchmarkRunGroup, benchmarkRunGroupIndex,
        benchmarkRunGroup.min);
  }

  private buildRunNumbersRow(values: string[]) {
    const runNumberRowElement = document.createElement('div');
    runNumberRowElement.className = 'run-numbers-row math-benchmark';

    for (let i = 0; i < values.length; i++) {
      const runNumberCellElement = document.createElement('div');
      runNumberCellElement.className = 'run-numbers-cell math-benchmark';
      runNumberCellElement.innerText = values[i];
      runNumberRowElement.appendChild(runNumberCellElement);
    }
    return runNumberRowElement;
  }

  private runBenchmarkSteps(
      chart: Chart, benchmarkRunGroup: BenchmarkRunGroup,
      benchmarkRunGroupIndex: number, step: number) {
    const runNumbersTable =
        this.querySelectorAll('.run-numbers-table')[benchmarkRunGroupIndex] as
        HTMLElement;
    if (step > benchmarkRunGroup.max ||
        this.stopMessages[benchmarkRunGroupIndex]) {
      this.stopMessages[benchmarkRunGroupIndex] = false;

      runNumbersTable.style.display = '';

      const canvas =
          this.querySelectorAll('.run-plot')[benchmarkRunGroupIndex] as
          HTMLCanvasElement;
      canvas.style.display = 'block';
      chart.update();

      const runMessage =
          this.querySelectorAll('.run-message')[benchmarkRunGroupIndex] as
          HTMLElement;
      runMessage.style.display = 'none';

      return;
    }

    const runNumberRowElement = document.createElement('div');
    runNumberRowElement.className = 'run-numbers-row math-benchmark';

    const rowValues: string[] = ['' + step];
    for (let i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
      const benchmarkRun = benchmarkRunGroup.benchmarkRuns[i];
      const benchmarkTest = benchmarkRun.benchmarkTest;

      const size = benchmarkRunGroup.stepToSizeTransformation != null ?
          benchmarkRunGroup.stepToSizeTransformation(step) :
          step;

      let resultString: string;
      let logString: string;
      let time = 0;
      let success = true;

      try {
        time = benchmarkTest.run(size);
        resultString = time.toFixed(3) + 'ms';
        logString = resultString;
      } catch (e) {
        success = false;
        resultString = 'Error';
        logString = e.message;
      }

      if (time >= 0) {
        if (success) {
          benchmarkRun.chartData.push({x: step, y: time});
        }
        rowValues.push(resultString);
      }
      console.log(benchmarkRun.name + '[' + size + ']: ' + logString);
    }
    runNumbersTable.appendChild(this.buildRunNumbersRow(rowValues));

    step += benchmarkRunGroup.stepSize;
    // Allow the UI to update.
    setTimeout(
        () => this.runBenchmarkSteps(
            chart, benchmarkRunGroup, benchmarkRunGroupIndex, step),
        100);
  }
}
document.registerElement(MathBenchmark.prototype.is, MathBenchmark);
