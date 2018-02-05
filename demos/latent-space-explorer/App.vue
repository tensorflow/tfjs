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
  <demo-header name="Latent Space Explorer"></demo-header>
  <div class="app">
    <!-- Presets -->
    <div class="presets">
      <div class="header sticky">
        <h3>Saved Samples</h3>
        <FontChooser
          :model="model"
          :selectedSample="selectedSample"
          v-on:select="changeSelected"
        />
        <div class="description">This demo allows you to interactively explore
           a 40-dimensional latent space of typefaces. The model used here
           was created by <a href="https://twitter.com/bengiswex">James Wexler</a>,
           based on work by
           <a href="https://erikbern.com/2016/01/21/analyzing-50k-fonts-using-deep-neural-networks.html">Erik Bernhardsson</a>,
           and is used in the distill.pub article
           <a href="https://distill.pub/2017/aia/">Using Artificial Intelligence to Augment Human Intelligence"</a>,
           which describes the model in greater detail.
        </div>
      </div>
    </div>
    <!-- Dimensions -->
    <div class="input">
      <div class="header">
        <h3>Basis Dimensions of the Latent Space</h3>
      </div>
      <div ref="loading">Loading...</div>
      <div ref="basis" class="basis">
        <BasisDimensions
          :scrollY="scrollY"
          :model="model"
          :modelData="modelData"
          :numSamples="numSamples"
          :selectedSample="selectedSample"
          :range="range"
          :vals="dimSliderVals"
          :width="width"
          v-on:select="changeSelected"
        />
      </div>
    </div>
    <!-- Output -->
    <div class="output">
      <div class="header sticky">
        <h3>Current Sample</h3>
        <Alphabet
          :model="model"
          :sample="selectedSample"
        />
      </div>
    </div>
  </div>
  <demo-footer></demo-footer>
</div>
</template>

<script>
import Vue from 'vue';

import DemoFooter from '../footer.vue';
import DemoHeader from '../header.vue';

import BasisDimensions from './components/BasisDimensions.vue';
import FontChooser from './components/FontChooser.vue';
import Alphabet from './components/Alphabet.vue';
import {FontModel} from './utils/FontModel';
import {Tensor1D} from 'deeplearn';

export default {
  components: {
    Alphabet,
    BasisDimensions,
    DemoFooter,
    DemoHeader,
    FontChooser,
  },
  data() {
    return {
      model: undefined,
      modelData: "A",
      numSamples: 9,
      range: 0.4,
      width: 400,
      dimSliderVals: [],
      selectedSample: undefined,
      scrollY: 0
    }
  },
  mounted: function() {
    window.addEventListener("resize", this.onresize);
    window.addEventListener("scroll", this.onscroll);

    let fonts = new FontModel();
    fonts.load(() => {
      this.$refs.loading.remove();
      this.model = fonts;
      this.range = fonts.range;
      this.resize();
    });
  },
  beforeDestroy: function() {
    window.removeEventListener("resize", this.onresize);
    window.removeEventListener("scroll", this.onscroll);
  },
  methods: {
    resize: function() {
      const width = this.$refs.basis.getBoundingClientRect().width;
      this.width = width;
      this.onscroll();
    },
    onscroll: function() {
      const y = window.scrollY;
      this.scrollY = Math.round(y / 20) * 20;
    },
    changeSelected: function(event) {
      // If this is the initial (default) selection and a URL hash was
      // provided then use the sample from the hash.

      if (event.isInitialSelection && window.location.hash) {
        this.parseUrlHash();
      } else {
        this.selectedSample = event.selectedSample;
        this.updateHash();
      }
    },
    updateHash: function() {
      if (this.selectedSample) {
        this.selectedSample.data().then(vals => {
          const hashStr = '#' + Array.from(vals)
            .map(val => parseFloat(val).toFixed(3))
            .join(',');
        history.replaceState(undefined, undefined, hashStr);
        });
      }
    },
    parseUrlHash: function() {
      const hash = window.location.hash;
      const dimVals = hash.substring(1).split(',').map(val => +val);
      // Set the selected sample and initial dimension slider values based
      // on the provided URL hash.
      this.dimSliderVals = dimVals;
      this.selectedSample = Tensor1D.new(dimVals);
    }
  }

}
</script>

<style scoped>
h3 {
  font-weight: 600;
  font-size: 18px;
  margin-top: 20px;
  border-top: 2px solid black;
  padding-top: 20px;
  line-height: 1em;
}
.app {
  line-height: 1.5em;
  max-width: 1200px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 1fr 3fr 2fr;
  grid-column-gap: 60px;
  position: relative;
  padding: 0 20px;
}
.basis {
  overflow: hidden;
}
.header {
  padding-top: 20px;
}
.description {
  border-top: solid 1px rgba(0, 0, 0, 0.1);
  margin-top: 20px;
  padding-top: 20px;
  font-size: 14px;
}
.sticky {
  position: -webkit-sticky;
  position: sticky;
  z-index: 100;
  top: 0;
}
.sliderlabel {
  font-size: 14px;
}
.slidervalue {
  border: none;
  font-size: 14px;
}
</style>
