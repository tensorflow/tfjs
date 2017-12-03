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
<div ref="container" class="container" v-on:mousedown="dragStart">
  <div
    class="tray"
    :style="{height: height + 'px'}">
      <Sample
        v-for="(sample, index) in samples"
        v-bind:key="index"
        :style="{position: 'absolute', left: sample.position + 'px'}"
        :visible="visible"
        :displayWidth="sampleWidth"
        :displayHeight="sampleWidth"
        :model="model"
        :character="modelData"
        :sample="sample.sample"
      />
    <div
      ref="selectedReticle"
      class="reticle selected"
      :style="{left: selectedX + 'px', height: height + 6 + 'px'}">
      <div class="label">{{format(selectedValue)}}</div>
    </div>
  </div>
  <Axis :min="extent[0]" :max="extent[1]" :width="width"/>
</div>
</template>

<script>
import Sample from './Sample.vue';
import Axis from './XAxis.vue';

import utils from '../utils/Utils';
import {range} from 'd3-array';
import {format} from 'd3-format';
import {scaleLinear, scaleBand} from 'd3-scale';
import {Scalar, Array1D, NDArrayMathCPU} from 'deeplearn';

const math = new NDArrayMathCPU(false);

export default {
  components: {Sample, Axis},
  data() {
    return {
      interpolate: scaleLinear(),
      position: scaleLinear(),
      bandScale: scaleBand(),
      formatter: format(",.3f"),
      visible: false,
      samples: [],
      offset: 0
    }
  },
  props: {
    modelData: { type: String, default: "A" },
    model: { },
    numSamples: { type: Number, default: 9 },
    width: { type: Number, default: 200 },
    initialValue: { type: Number, default: 1},
    selectedSample: { default: () => {[]}},
    direction: { default: () => {[]}},
    range: { type: Number, default: 1 },
    scrollY: {default: 0}
  },
  computed: {
    extent: function() { return [-this.range, this.range]; },
    dimensions: function() { return this.model ? this.model.dimensions : 0; },
    zero: function() { return Array1D.zeros([this.dimensions]); },
    hoverScale: function() {
      return this.interpolate.domain([0, this.width]).range(
        this.extent).clamp(true);
    },
    bands: function() {
      return this.bandScale.domain(range(this.numSamples)).range(
        [0, this.width]);
    },
    pos: function() {
      return this.position.domain([0, this.numSamples - 1]).range(this.extent);
    },
    sampleWidth: function() { return this.bands.bandwidth(); },
    height: function() { return this.sampleWidth; },
    unitDirection: function() {
      let length = math.sum(math.multiply(this.direction, this.direction));
      return math.divide(this.direction, length);
    },
    selectedValue: function() {
      const scalar = math.dotProduct(this.unitDirection, this.selectedSample)
      return scalar.getValues()[0]
    },
    selectedX: function() {
      return this.hoverScale.invert(this.selectedValue);
    },

  },
  mounted() {
    this.recomputeSamples();
    this.checkVisibility();
  },
  watch: {
    selectedSample: function(value) { this.recomputeSamples(); },
    model: function() { this.recomputeSamples(); },
    selectedValue: function() { this.recomputeSamples(); },
    width: function() { this.recomputeSamples(); },
    pos: function() { this.recomputeSamples(); },
    bands: function() { this.recomputeSamples(); },
    scrollY: function(val) { this.checkVisibility(); }
  },
  methods: {
    dragStart: function(event) {
      document.addEventListener("mousemove", this.dragUpdate);
      document.addEventListener("mouseup", this.dragEnd);
      this.offset = this.$refs.container.getBoundingClientRect().left;
    },
    dragUpdate: function(event) {
      event.preventDefault();
      this.select(event.pageX - this.offset);
    },
    dragEnd: function(event) {
      this.select(event.pageX - this.offset);
      document.removeEventListener("mouseup", this.dragEnd);
      document.removeEventListener("mousemove", this.dragUpdate);
    },
    recomputeSamples: function() {
      let samples = [];
      for (var i = 0; i < this.numSamples; i++) {
        let delta = math.sub(
          Scalar.new(this.pos(i)), Scalar.new(this.selectedValue));
        let newSample = math.add(math.multiply(
          this.unitDirection, delta), this.selectedSample);
        samples.push({
          sample: newSample,
          position: this.bands(samples.length)
        });
      }
      this.samples = samples;
    },
    checkVisibility: function() {
      const buffer = 200;
      const top = this.$refs.container.getBoundingClientRect().top;
      const visible = (top > -buffer  && top < window.innerHeight + buffer);
      this.visible = visible;
    },
    format: function(val) {
      return this.formatter(val);
    },
    select: function(x) {
      const value = this.hoverScale(x)
      let delta = math.sub(Scalar.new(value), Scalar.new(this.selectedValue));
      let newSample = math.add(math.multiply(
        this.unitDirection, delta), this.selectedSample);
      this.$emit("select", {selectedSample: newSample});
    }
  }
}
</script>

<style scoped>
div.container {
  width: 100%;
}
div.tray {
  position: relative;
  width: 100%;
}
div.mouse {
  position: absolute;
  height: 60px;
  width: 100%;
  background: white
}
div.tray > * {
  pointer-events: none;
}
.reticle {
  height: calc(100% + 6px);
  position: absolute;
  width: 3px;
  border-radius: 2px;
  background-color: rgb(255, 152, 0);
  z-index: 1;
}
.reticle .label {
  background-color: rgb(255, 152, 0);
  color: white;
  padding: 2px;
  border-radius: 4px;
  font-size: 9px;
  line-height: 9px;
  text-align: center;
  width: 30px;
  position: absolute;
  left: -16px;
  bottom: -13px;
}
.hover.reticle {
  border-color:rgba(0, 0, 0, 0.3);
}
.selected.reticle {
  border-color: rgba(0, 0, 0, 0.8);
}
</style>
