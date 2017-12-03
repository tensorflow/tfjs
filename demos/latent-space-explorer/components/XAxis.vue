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
<div class="domain">
  <div v-for="tick in ticks" class="tick" :style="{left: tick.position + 'px'}">
    <span class="label">{{tick.value}}</span>
  </div>
</div>
</template>

<script>
import {scaleLinear} from "d3-scale";

export default {
  data() {
    return {
      scale: scaleLinear()
    }
  },
  props: {
    min: { default: 0 },
    max: { default: 1 },
    width: { default: 1 },
  },
  computed: {
    xScale: function() {
      return this.scale.domain([this.min, this.max]).range([0, this.width]);
    },
    ticks: function() {
      return this.xScale.ticks().map(
        t => { return {value: t, position: this.xScale(t)}; });
    }
  }
}
</script>

<style scoped>
.domain {
  position: relative;
  height: 18px;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  margin-bottom: 12px;
}

.tick {
  position: absolute;
  border-left: 1px solid rgba(0, 0, 0, 0.2);
  height: 4px;
}

.tick .label {
  text-align: center;
  position: absolute;
  color: rgba(0, 0, 0, 0.3);
  font-size: 9px;
  line-height: 9px;
  width: 10px;
  left: -5px;
  top: 8px;
  margin: 0;
  padding: 0;
}
</style>
