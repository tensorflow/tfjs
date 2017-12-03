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
<div
  :style="{width: displayWidth + 'px', height: displayHeight + 'px'}"
>
  <canvas
    ref="canvas"
    :width="width"
    :height="height"
    :class="{ waiting: waiting || dirty, ready: !waiting && !dirty }"
  ></canvas>
</div>
</template>

<script>
export default {
  data() {
    return {
      width: 64,
      height: 64,
      waiting: false,
      dirty: false,
      priority: 0,
      position: 0,
      id: Math.random()
    }
  },
  props: {
    modelData: { type: String, default: "A" },
    displayWidth: { type: Number, default: "64" },
    displayHeight: { type: Number, default: "64" },
    sample: { default: () => {[]}},
    model: { },
    visible: {default: false}
  },
  mounted: function() {
    this.render();
  },
  watch: {
    visible: function(val) {
      if (this.model && this.sample) {
        if (this.dirty && val) {
          this.render();
        }
      }
    },
    sample: function(newValue, oldValue) {
      if(newValue && newValue !== oldValue) {
        if (this.visible) {
          this.render();
        } else {
          this.dirty = true;
        }
      }
    },
    model: function(val) {
      this.render();
    }
  },
  methods: {
    measure: function() {
      const bb = this.$refs.canvas.getBoundingClientRect();
      const screen = window.innerHeight;
      this.position = bb.top;
      this.visible = (bb.top < (window.innerHeight) && bb.bottom > 0)
    },
    render: function() {
      this.waiting = true;
      if (this.model) {this.model.metaData = this.modelData;}
      if (this.model && this.sample && this.sample.size) {
        let canvas = this.$refs.canvas;
        let canvasContext = canvas.getContext("2d");
        this.model.get(this.id, [this.sample, canvasContext])
          .then(() => {
            this.waiting = false;
            this.dirty = false;
          });
      }
    }
  }
}

</script>

<style scoped>
div {
  display: inline-block;
  position: relative;
}
.waiting {
  opacity: 0.2;
}
canvas {
  position: absolute;
  width: 100%;
  height: 100%;
  mix-blend-mode: multiply;
}
</style>
