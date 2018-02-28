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

// tslint:disable-next-line:max-line-length
import {buildWeightMap, loadRemoteProtoFile, loadRemoteWeightFile, tensorflow} from '../data/index';
import {TensorMap} from '../data/index';
import {OperationMapper} from '../operations/index';

import {GraphExecutor} from './graph_executor';

export class Model {
  private executor: GraphExecutor;
  constructor(private modelUrl: string, private weightUrl: string) {}

  load(): Promise<boolean> {
    const graphPromise = loadRemoteProtoFile(this.modelUrl);
    const weightPromise = loadRemoteWeightFile(this.weightUrl);
    const weightMapPromise = buildWeightMap(graphPromise, weightPromise);
    const executorPromise = graphPromise.then(
        graph => this.executor =
            new GraphExecutor(OperationMapper.Instance.transformGraph(graph)));

    return Promise.all([weightMapPromise, executorPromise])
        .then(([weightMap, executor]) => {
          executor.weightMap = weightMap;
          return true;
        });
  }

  predict(inputs: TensorMap): TensorMap {
    return this.executor.execute(inputs);
  }
}
