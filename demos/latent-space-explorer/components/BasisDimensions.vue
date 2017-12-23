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
  <div ref="root" class="root" v-for="(basis, i) in basisDimensions">
    <h4 class="label">
      <span>{{i + 1}}</span>
    </h4>
    <Tray
      :scrollY="scrollY"
      :model="model"
      :modelData="modelData"
      :direction="basis"
      :range="range"
      :numSamples="numSamples"
      :selectedSample="selectedSample"
      :initialValue="vals[i]"
      :width="width - 50"
      v-on:select="select"
    />
  </div>
</div>
</template>

<script>
import Sample from './Sample.vue';
import Tray from './Tray.vue';
import {range} from 'd3-array';
import {Array1D, ENV} from 'deeplearn';

const math = ENV.math;

export default {
  components: {Sample, Tray},
  data() {
    return {
      basisDimensions: []
    };
  },
  watch: {
    model: function(m) {
      const dims = m ? m.dimensions: 0;
      this.basisDimensions = range(dims).map(dim => {
        return math.oneHot(Array1D.new([dim]), dims).as1D();
      });
    }
  },
  props: {
    modelData: { type: String, default: "A" },
    selectedSample: { default: () => {[]}},
    model: { },
    width: { type: Number, default: 300},
    numSamples: { type: Number, default: 5 },
    range: { type: Number, default: 1 },
    vals: { type: Array, default: function() { return []; }},
    scrollY: {default: 0}
  },
  methods: {
    select: function(event) {
      this.$emit("select", event);
    }
  }
}
</script>

<style scoped>
.root {
  margin-top: 8px;
  padding-right: 8px;
  display: grid;
  grid-template-columns: 40px 1fr;
  margin-bottom: 30px;
}
h4.label {
  margin: 10px 0 10px 0;
  font-weight: 400;
  font-size: 14px;
}

.label span {
  color: rgba(0, 0, 0, 0.6);
  border: 1px solid rgba(0, 0, 0, 0.3);
  padding: 6px;
  border-radius: 50%;
  font-weight: 300;
  font-size: 11px;
  display: block;
  width: 12px;
  height: 12px;
  line-height: 1.2em;
  text-align: center;
}
</style>
