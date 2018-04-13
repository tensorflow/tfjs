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
export let MathBenchmarkPolymer: new () => PolymerHTMLElement =
    PolymerElement({is: 'math-benchmark', properties: {benchmarks: Array}});

export class MathBenchmark extends MathBenchmarkPolymer {
  // Polymer properties.
  benchmarks: BenchmarkRunGroup[];
  stopMessages: boolean[];

  ready() {
    const groups = getRunGroups();
    // Set up the benchmarks UI.

    const benchmarks: BenchmarkRunGroup[] = [];
    this.stopMessages = [];
    groups.forEach(group => {
      if (group.selectedOption == null) {
        group.selectedOption = '';
      }
      benchmarks.push(group);
      this.stopMessages.push(false);
    });
    this.benchmarks = benchmarks;

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

  getDisplayParams(paramsMap: {[option: string]: {}}, selectedOption: string):
      string {
    const params = paramsMap[selectedOption];
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

  private runBenchmarkGroup(
      groups: BenchmarkRunGroup[], benchmarkRunGroupIndex: number) {
    const benchmarkRunGroup = groups[benchmarkRunGroupIndex];

    const canvas = this.querySelectorAll('.run-plot')[benchmarkRunGroupIndex] as
        HTMLCanvasElement;
    // Avoid to growing size of rendered chart.
    canvas.width = 360;
    canvas.height = 270;
    const context = canvas.getContext('2d') as CanvasRenderingContext2D;

    const datasets: ChartDataSets[] = [];
    for (let i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
      benchmarkRunGroup.benchmarkRuns[i].clearChartData();
      const hue = Math.floor(360 * i / benchmarkRunGroup.benchmarkRuns.length);
      datasets.push({
        data: benchmarkRunGroup.benchmarkRuns[i].chartData,
        fill: false,
        label: benchmarkRunGroup.benchmarkRuns[i].name,
        borderColor: `hsl(${hue}, 100%, 40%)`,
        backgroundColor: `hsl(${hue}, 100%, 70%)`,
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
                return `${label}ms`;
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

  private async runBenchmarkSteps(
      chart: Chart, runGroup: BenchmarkRunGroup, benchmarkRunGroupIndex: number,
      step: number) {
    const runNumbersTable =
        this.querySelectorAll('.run-numbers-table')[benchmarkRunGroupIndex] as
        HTMLElement;
    if (step > runGroup.max || this.stopMessages[benchmarkRunGroupIndex]) {
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

    const rowValues: string[] = [step.toString()];
    for (let i = 0; i < runGroup.benchmarkRuns.length; i++) {
      const run = runGroup.benchmarkRuns[i];
      const test = run.benchmarkTest;

      const size = runGroup.stepToSizeTransformation != null ?
          runGroup.stepToSizeTransformation(step) :
          step;

      const opType = runGroup.selectedOption;
      const time = await test.run(size, opType, runGroup.params[opType]);
      const resultString = time.toFixed(3) + 'ms';

      if (time >= 0) {
        run.chartData.push({x: step, y: time});
        rowValues.push(resultString);
      }
      console.log(`${run.name}[${size}]: ${resultString}`);
    }
    runNumbersTable.appendChild(this.buildRunNumbersRow(rowValues));

    step += runGroup.stepSize;
    // Allow the UI to update.
    setTimeout(
        () => this.runBenchmarkSteps(
            chart, runGroup, benchmarkRunGroupIndex, step),
        100);
  }
}
document.registerElement(MathBenchmark.prototype.is, MathBenchmark);
