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
  <div v-for="(sample, index) in samples" class="typeface" >
    <div v-for="char in letters" class="character" v-on:click="select(sample)">
      <Sample
        :displayWidth="width"
        :displayHeight="width"
        :model="model"
        :modelData="char"
        :sample="sample"
        :visible="visible"
      />
    </div>
    <button v-on:click="deleteSample(index)">✕</button>
    <span class="ellipse">…</span>
  </div>
  <button v-on:click="save">Save current sample</button>
</div>
</template>

<script>
import Sample from './Sample.vue';
import {
  zero, serif, serifBold, serifLight, sansLight, crispSerif, dotMatrix, casual,
  serifBlackItalic, serifItalic, square
  } from '../utils/FontExamples';

export default {
  components: {
    Sample
  },
  data() {
    return {
      width: 20,
      visible: true,
      letters: "ABCDEFG".split("")
    }
  },
  props: {
    modelData: { type: String, default: "A" },
    selectedSample: { },
    model: { },
    samples: { type: Array, default: () => [
      zero, crispSerif, serifItalic, serifBlackItalic, sansLight, casual,
      dotMatrix] }
  },
  watch: {
    model: function(val) {
      this.select(this.samples[0], true);
    }
  },
  methods: {
    deleteSample: function(index) {
      this.samples.splice(index, 1);
    },
    select: function(sample, isInitialSelection) {
      this.$emit("select", {selectedSample: sample, isInitialSelection});
    },
    save: function() {
      this.samples.push(this.selectedSample);
    }
  }
}
</script>

<style scoped>
.typeface {
  position: relative;
  cursor: pointer;
  opacity: 0.5;
  margin: 3px 0;
  padding: 3px 0;
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  line-height: 17px;
}
.typeface button {
  position: absolute;
  top: -17px;
  right: 0;
  display: none;
  border-radius: 50%;
  width: 1.6em;
  height: 1.6em;
  background: rgba(0, 0, 0, 0.1);
  color: white;
  padding: 0;
  text-align: center;
  line-height: 1.5em;
  cursor: pointer;
  border: none;
}
.typeface:hover button {
  display: block;
}
.typeface button:hover {
  background: red;
}
.typeface:hover {
  opacity: 1;
}
button {
  margin-top: 20px;
}
.ellipse {
  position: relative;
  top: -3px;
}
.typeface.selected {
  border-left: 3px solid hsl(24, 100%, 50%);
  padding-left: 18px;
  opacity: 1;
}
.character {
  display: inline-block;
  position: relative;
  width: 17px;
  height: 17px;
  overflow: hidden;
}
.character > div {
  position: relative;
  left: -2px;
}
</style>
