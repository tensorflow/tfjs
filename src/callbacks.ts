/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {CustomCallback, Logs, nextFrame, util} from '@tensorflow/tfjs';
import * as ProgressBar from 'progress';

// A helper class created for testing with the jasmine `spyOn` method, which
// operates only on member methods of objects.
// tslint:disable-next-line:no-any
export const progressBarHelper: {ProgressBar: any, log: Function} = {
  ProgressBar,
  log: console.log
};

/**
 * Terminal-based progress bar callback for tf.Model.fit().
 */
export class ProgbarLogger extends CustomCallback {
  private numTrainBatchesPerEpoch: number;
  private progressBar: ProgressBar;
  private currentEpochBegin: number;
  private epochDurationMillis: number;
  private usPerStep: number;

  /**
   * Construtor of LoggingCallback.
   */
  constructor() {
    super({
      onTrainBegin: async (logs?: Logs) => {
        const samples = this.params.samples as number;
        const batchSize = this.params.batchSize as number;
        util.assert(
            samples != null,
            'ProgbarLogger cannot operate when samples is undefined or null.');
        util.assert(
            batchSize != null,
            'ProgbarLogger cannot operate when batchSize is undefined or ' +
                'null.');
        this.numTrainBatchesPerEpoch = Math.ceil(samples / batchSize);
      },
      onEpochBegin: async (epoch: number, logs?: Logs) => {
        progressBarHelper.log(`Epoch ${epoch + 1} / ${this.params.epochs}`);
        this.currentEpochBegin = util.now();
      },
      onBatchEnd: async (batch: number, logs?: Logs) => {
        if (batch === 0) {
          this.progressBar = new progressBarHelper.ProgressBar(
              'eta=:eta :bar :placeholderForLossesAndMetrics',
              {total: this.numTrainBatchesPerEpoch + 1, head: `>`});
        }
        this.progressBar.tick({
          placeholderForLossesAndMetrics: this.formatLogsAsMetricsContent(logs)
        });
        await nextFrame();
        if (batch === this.numTrainBatchesPerEpoch - 1) {
          this.epochDurationMillis = util.now() - this.currentEpochBegin;
          this.usPerStep =
              this.epochDurationMillis / (this.params.samples as number) * 1e3;
        }
      },
      onEpochEnd: async (epoch: number, logs?: Logs) => {
        this.progressBar.tick({placeholderForLossesAndMetrics: ''});
        const lossesAndMetricsString = this.formatLogsAsMetricsContent(logs);
        progressBarHelper.log(
            `${this.epochDurationMillis.toFixed(0)}ms ` +
            `${this.usPerStep.toFixed(0)}us/step - ` +
            `${lossesAndMetricsString}`);
        await nextFrame();
      },
    });
  }

  private formatLogsAsMetricsContent(logs: Logs): string {
    let metricsContent = '';
    const keys = Object.keys(logs).sort();
    for (const key of keys) {
      if (this.isFieldRelevant(key)) {
        metricsContent += `${key}=${logs[key].toFixed(2)} `;
      }
    }
    return metricsContent;
  }

  private isFieldRelevant(key: string) {
    return key !== 'batch' && key !== 'size';
  }
}
