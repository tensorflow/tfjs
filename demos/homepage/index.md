---
layout: default
---
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
<!-- CPPN Demo Banner -->
<div class="banner-cover" id='banner'>
  <canvas id="inference"></canvas>
  <div class="mdl-grid banner">
    <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
    <div class="mdl-cell mdl-cell--6-col mdl-cell--8-col-tablet mdl-cell--4-col-phone banner-text">
      <div class="mdl-typography--display-4">deeplearn.js</div>
      <div class="mdl-typography--display-1" style="margin-left: 12px;">
        a hardware-accelerated <br/>
        machine intelligence<br/>
        library for the web
      </div>
    </div>
    <div class="mdl-layout-spacer  mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
    <div class="mdl-cell mdl-cell--3-col mdl-cell--8-col-tablet mdl-cell--4-col-phone cppn-controls">
      <div class="mdl-textfield mdl-js-textfield mdl-textfield--floating-label getmdl-select getmdl-select__fix-height">
        <input class="mdl-textfield__input" type="text" id="colormode" value="rgb" readonly tabIndex="-1">
        <label for="colormode">
          <i class="mdl-icon-toggle__label material-icons">keyboard_arrow_down</i>
        </label>
        <label for="colormode" class="mdl-textfield__label">Color mode</label>
        <ul for="colormode" class="mdl-menu mdl-menu--bottom-left mdl-js-menu" id="color-selector">
          <li class="mdl-menu__item" data-val="rgb">rgb</li>
          <li class="mdl-menu__item" data-val="rgba">rgba</li>
          <li class="mdl-menu__item" data-val="hsv">hsv</li>
          <li class="mdl-menu__item" data-val="hsva">hsva</li>
          <li class="mdl-menu__item" data-val="yuv">yuv</li>
          <li class="mdl-menu__item" data-val="yuva">yuva</li>
          <li class="mdl-menu__item" data-val="bw">bw</li>
        </ul>
      </div>
      <br>
      <div class="mdl-textfield mdl-js-textfield mdl-textfield--floating-label getmdl-select getmdl-select__fix-height">
        <input class="mdl-textfield__input" type="text" id="activation-fn" value="tanh" readonly tabIndex="-1">
        <label for="activation-fn">
          <i class="mdl-icon-toggle__label material-icons">keyboard_arrow_down</i>
        </label>
        <label for="activation-fn" class="mdl-textfield__label">Activation function</label>
        <ul for="activation-fn" class="mdl-menu mdl-menu--bottom-left mdl-js-menu" id="activation-selector">
          <li class="mdl-menu__item" data-val="tanh">tanh</li>
          <li class="mdl-menu__item" data-val="sin">sin</li>
          <li class="mdl-menu__item" data-val="relu">relu</li>
          <li class="mdl-menu__item" data-val="step">step</li>
        </ul>
      </div>
      <br>
      <div>Number of layers:
        <div id="layers-count" style="display: inline-block"></div>
      </div>
      <p style="width:200px">
        <input class="mdl-slider mdl-js-slider" type="range" min="0" max="7" value="2" tabindex="1" id="layers-slider">
      </p>
      <div>z1 time</div>
      <p style="width:200px">
        <input class="mdl-slider mdl-js-slider" type="range" min="1" max="100" value="1" tabindex="2" id="z1-slider">
      </p>
      <div>z2 time</div>
      <p style="width:200px">
        <input class="mdl-slider mdl-js-slider" type="range" min="1" max="100" value="1" tabindex="3" id="z2-slider">
      </p>
      <button class="mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--accent" type="button" id="random">randomize</button>
      <button class="mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--accent" type="button" id="toggle">stop</button>
      <br><br>
      <a href="http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/" target="_blank">What is a CPPN?</a>
      <div id="disabled-demo-overlay" style="display: none">
        <div id="disabled-demo">Unfortunately you do not have a WebGL-enabled device so you cannot see this demo.</div>
      </div>
    </div>
    <div class="mdl-layout-spacer"></div>
  </div>
</div>
<!-- Introduction Section -->
<div class= "mdl-grid intro-text">
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
  <div class= "mdl-cell mdl-cell--5-col mdl-cell--8-col-tablet mdl-cell-4-col-phone">
    <p class='intro-headline mdl-typography--headline'><span class="deeplearn-shine">deeplearn.js</span> is an open-source library that brings performant machine learning building blocks to the web, allowing you to train neural networks in a browser or run pre-trained models in inference mode.</p>
  </div>
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
  <div class= "intro-text mdl-cell mdl-cell--4-col mdl-cell--8-col-tablet mdl-cell-4-col-phone">
    <p class='intro-body mdl-typography--body-1'>We provide two APIs, an
    <span class="deeplearn-shine">immediate execution model</span> (think NumPy)
    and a <span class="deeplearn-shine">deferred execution model</span>
    mirroring the TensorFlow API.<br><br><span class="deeplearn-shine">deeplearn.js</span>
    was originally developed by the Google Brain PAIR team to build powerful
    interactive machine learning tools for the browser. You can use the library
    for everything from education, to model understanding, to art projects.</p>
  </div>
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
</div>
<!-- Demo Section -->
<div class="examples" id='demos'>
  <div class="section-title mdl-grid">
    <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
    <p class='mdl-typography--display-2 mdl-cell mdl-cell--9-col'>Examples</p>
    <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
  </div>
  <!-- Featured Card -->
  <div class='featured-demo mdl-grid'>
    <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
    <div class="demo-card mdl-card mdl-shadow--4dp feature-card mdl-cell mdl-cell--9-col">
      <a href="demos/model-builder/model-builder-demo.html">
        <div class='mdl-card__title' id="model-builder">
          <h1 class= 'mdl-card__title-text'>Model Builder</h1>
        </div>
      </a>
      <div class= 'mdl-card__supporting-text'>Build a neural network in your browser, without code!</div>
      <div class='mdl-card__actions mdl-card--border'>
        <a href="demos/model-builder/model-builder-demo.html">
          <button class="mdl-button mdl-button--raised mdl-button--colored mdl-js-button mdl-button--primary mdl-js-ripple-effect">
            Demo
          </button>
        </a>
        <a href="https://github.com/PAIR-code/deeplearnjs/tree/master/demos/model-builder/">
          <button class="mdl-button mdl-button--raised mdl-button--colored mdl-js-button mdl-button--primary mdl-js-ripple-effect">
            Code
          </button>
        </a>
      </div>
    </div>
    <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
  </div>
  <!-- Demo Carousel -->
  <div class="demo-carousel mdl-grid">
    <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
    <div class="demo-card mdl-card mdl-shadow--2dp square-card mdl-cell mdl-cell--3-col mdl-cell--8-col-tablet mdl-cell--4-col-phone">
      <a href="demos/imagenet/imagenet-demo.html">
        <div class='mdl-card__title' id="webcam">
          <span class='mdl-card__title-text'>Webcam Imagenet</span>
        </div>
      </a>
      <div class='mdl-card__supporting-text'>Squeezenet running in the browser</div>
      <div class='mdl-card__actions mdl-card--border'>
        <a href="demos/imagenet/imagenet-demo.html">
          <button class="mdl-button mdl-button--raised mdl-button--colored mdl-js-button mdl-button--primary mdl-js-ripple-effect">
            Demo
          </button>
        </a>
        <a href="https://github.com/PAIR-code/deeplearnjs/tree/master/demos/imagenet">
          <button class="mdl-button mdl-button--raised mdl-button--colored mdl-js-button mdl-button--primary mdl-js-ripple-effect">
            Code
          </button>
        </a>
      </div>
    </div>
    <div class="demo-card mdl-card mdl-shadow--2dp square-card mdl-cell mdl-cell--3-col mdl-cell--8-col-tablet mdl-cell--4-col-phone ">
      <a href="demos/nn-art/nn-art-demo.html">
        <div class='mdl-card__title' id="nnart">
          <span class= 'mdl-card__title-text'>NNArt</span>
        </div>
      </a>
      <div class='mdl-card__supporting-text'>Play with an animating CPPN</div>
      <div class='mdl-card__actions mdl-card--border'>
        <a href="demos/nn-art/nn-art-demo.html">
          <button class="mdl-button mdl-button--raised mdl-button--colored mdl-js-button mdl-button--primary mdl-js-ripple-effect">
            Demo
          </button>
        </a>
        <a href="https://github.com/PAIR-code/deeplearnjs/tree/master/demos/nn-art">
          <button class="mdl-button mdl-button--raised mdl-button--colored mdl-js-button mdl-button--primary mdl-js-ripple-effect">
            Code
          </button>
        </a>
      </div>
    </div>
    <div class="demo-card mdl-card mdl-shadow--2dp square-card mdl-cell mdl-cell--3-col mdl-cell--8-col-tablet mdl-cell--4-col-phone">
      <a href="demos/benchmarks/benchmark-demo.html">
        <div class='mdl-card__title' id="benchmarks">
          <span class= 'mdl-card__title-text'>Benchmarks</span>
        </div>
      </a>
      <div class='mdl-card__supporting-text'>Test the library's performance</div>
      <div class='mdl-card__actions mdl-card--border'>
        <a href="demos/benchmarks/benchmark-demo.html">
          <button class="mdl-button mdl-button--raised mdl-button--colored mdl-js-button mdl-button--primary mdl-js-ripple-effect">
            Demo
          </button>
        </a>
        <a href=" https://github.com/PAIR-code/deeplearnjs/tree/master/demos/benchmarks">
          <button class="mdl-button mdl-button--raised mdl-button--colored mdl-js-button mdl-button--primary mdl-js-ripple-effect">
            Code
          </button>
        </a>
      </div>
    </div>
    <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
  </div>
</div>


<div class='mdl-grid'>
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
  <div class="mdl-cell mdl-cell--10-col">
    <h2>Ready to Get Started?</h2>
    {% assign default_paths = site.pages | map: "path" %}
    {% assign page_paths = site.header_pages | default: default_paths %}
    <ul class="index">
      {% for path in default_paths %}
        {% assign my_page = site.pages | where: "path", path | first %}
        {% assign title = my_page.title | trim %}
        {% if title %}
        <li>
          <a href="{{ my_page.url | relative_url }}">
            {{my_page.title | escape }}
          </a>
        </li>
        {% endif %}
      {% endfor %}
    </ul>
  </div>
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
</div>


<div class='mdl-grid'>
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
  <div class="mdl-cell mdl-cell--10-col">
    <h2 class= 'mdl-card__title-text'>Acknowledgements</h2>
  </div>
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
</div>

<div class= "mdl-grid ack">
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
  <div class= "mdl-cell mdl-cell--5-col mdl-cell--8-col-tablet mdl-cell-4-col-phone">
    <p class="intro-body mdl-typography--body-1">
      <span class="deeplearn-shine">deeplearn.js</span> was originally developed by Nikhil Thorat, Daniel Smilkov and Charles Nicholson.
    </p>
  </div>
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
  <div class= "ack mdl-cell mdl-cell--4-col mdl-cell--8-col-tablet mdl-cell-4-col-phone">
    <p class='intro-body mdl-typography--body-1'>
      We would like to acknowledge Chi Zeng, David Farhi, Mahima Pushkarna,
      Lauren Hannah-Murphy, Minsuk (Brian) Kahng, James Wexler, Martin Wattenberg,
      Fernanda Viégas, Greg Corrado, Jeff Dean for their tremendous help, and
      Google Brain PAIR for providing support for the project.
    </p>
  </div>
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
</div>
<script src="bundle.js"></script>
