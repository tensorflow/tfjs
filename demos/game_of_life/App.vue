<!-- Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================-->
<template>
  <div>
    <graph-source/>
    <demo-header name="Game-of-Life deep-learning demo"></demo-header>
    <div id="app">
      <div class="train">
        <table class="train-controls">
          <tr>
            <td>Board size:</td>
            <td>
              <input type="text" v-model="boardSize" :disabled="running">
            </td>
          </tr>
          <tr>
            <td>Training steps:</td>
            <td>
              <input type="text" v-model="trainingSteps" :disabled="running">
            </td>
          </tr>
          <tr>
            <td>Training batch size:</td>
            <td>
              <input type="text" v-model="trainingBatchSize" :disabled="running">
            </td>
          </tr>
          <tr>
            <td>Learning rate:</td>
            <td>
              <input type="text" v-model="learningRate" :disabled="running">
            </td>
          </tr>
          <tr>
            <td>Number of layers:</td>
            <td>
              <input type="text" v-model="numLayers" :disabled="running">
            </td>
          </tr>
          <tr>
            <td>Update Interval:</td>
            <td>
              <input type="text" v-model="updateInterval">
            </td>
          </tr>
          <tr>
            <td>Use Log Cost:</td>
            <td>
              <input type="checkbox" v-model="useLogCost" :disabled="running">
            </td>
          </tr>
          <tr>
            <td colspan="2" class="buttons">
              <button v-on:click="onAddSequenceClicked">Add Sequence</button>
              <button v-on:click="onTrainModelClicked" :disabled="running">Train Model</button>
              <button v-on:click="onResetButtonClicked" :disabled="running">Reset</button>
            </td>
          </tr>
        </table>

        <div class="train-console">
          <div>
            <span class="train-display">Waiting to train.</span>
            <span class="data-display"></span>
          </div>
          <div class="train-graph">
            <canvas id="myChart" width="600" height="200"></canvas>
          </div>
        </div>

        <div class="clearfix"></div>
      </div>

      <div class="worlds-display"></div>
    </div>
    <demo-footer></demo-footer>
  </div>
</template>

<script lang="ts" src="./app.ts"></script>

<style>
.world {
  display: inline-block;
}

.world + .world {
  margin-left: 20px;
}

.world-display + .world-display {
  margin-top: 10px;
}

.board {
  background-color: #cccccc;
  display: inline-block;
  padding: 5px;
}

.row {
  line-height: 10px;
}

.column {
  min-height: 20px;
  min-width: 20px;
  display: inline-block;
  margin: 1px;
}

.column.alive {
  background-color: #333333;
}

.column.dead {
  background-color: #ffffff;
}

.worlds-display {
  text-align: center;
  padding-top: 10px;
}

.train {
  margin-left: auto;
  margin-right: auto;
  max-width: 925px;
  border-bottom: 1px dashed #333333;
  padding-top: 20px;
}

.train-controls {
  float: left;
}

.train-console {
  float: right;
}

.clearfix {
  clear: both;
}

.buttons {
  padding: 15px 0;
}

.train-display {
  color: #333333;
  text-decoration: underline;
}
</style>
