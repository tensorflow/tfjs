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
  <div v-for="section in sections" class="section">
    <div v-for="char in section" :key="char" class="character">
      <Sample
        :displayWidth="width"
        :displayHeight="width"
        :model="model"
        :modelData="char"
        :sample="sample"
        :visible="visible"
        on:select="fire('select', event)"
      />
    </div>
  </div>
</div>
</template>

<script>
import Sample from "./Sample.vue";

export default {
  components: {
    Sample
  },
  data() {
    return {
      width: 48,
      visible: true,
      sections:[
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("")
      ]
    }
  },
  props: {
    modelData: { type: String, default: "A" },
    model: { },
    sample: { default: () => {[]}},
  },
}
</script>

<style scoped>
.character {
  display: inline;
}
.section:first-child {
  border-bottom: solid 1px rgba(0, 0, 0, 0.1);
  padding-bottom: 20px;
  margin-bottom: 20px;
}
</style>
