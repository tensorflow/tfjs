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
<style>
.mdl-layout__header-row {
  padding-left: 30px;
}
a.main-title {
  color: rgb(66,66,66);
  text-decoration: none;
}
</style>

<script src="./header.ts"></script>

<template>
<header class="mdl-layout__header mdl-layout--fixed-header is-casting-shadow">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700" type="text/css">
  <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
  <link rel="stylesheet" href="https://code.getmdl.io/1.3.0/material.orange-deep_purple.min.css" />
  <div class="mdl-layout__header-row">
    <!-- Title -->
    <a class="main-title" href="/"><span class="mdl-layout-title">deeplearn.js</span></a>&nbsp;&nbsp;<span>{{name}}</span>
    <!-- Add spacer, to align navigation to the right -->
    <div class="mdl-layout-spacer"></div>
    <!-- Navigation. We hide it in small screens. -->
    <nav class="mdl-navigation mdl-layout--large-screen-only">
      <a class="mdl-navigation__link" target='_blank' href="https://github.com/PAIR-code/deeplearnjs">Code</a>
      <a class="mdl-navigation__link" href="/#getting-started">Getting Started</a>
      <a class="mdl-navigation__link" href="/#demos">Examples</a>
      <a class="mdl-navigation__link" href="/docs/tutorials/index.html">Tutorials</a>
      <a class="mdl-navigation__link" href="/docs/api/globals.html">API Reference</a>
    </nav>
  </div>
</header>
</template>
