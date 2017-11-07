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
        <div id="disabled-demo">
          Your device is not yet supported, so we cannot show this demo. We are working hard on supporting more devices. For now, come back on desktop Chrome!
        </div>
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
<div class="examples" id="demos">
  <div class="section-title mdl-grid">
    <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
    <p class="mdl-typography--display-2 mdl-cell mdl-cell--12-col">Examples</p>
    <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
  </div>

  <div class="featured-demo mdl-grid">
    <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
    <div class="demo-card featured-demo mdl-card mdl-shadow--4dp feature-card mdl-cell mdl-cell--12-col">
      <a href="demos/playground/examples.html" style="overflow: hidden">
        <div class="mdl-card__title" id="playground">
          <h1 class="mdl-card__title-text" class="mdl-card__title-text">Interactive Playground</h1>
        </div>
      </a>
      <div class="mdl-card__supporting-text">Play with deeplearn.js code in the browser with no installs</div>
      <div class="mdl-card__actions mdl-card--border">
        <a href="demos/playground/examples.html">
          <button class="mdl-button mdl-button--raised mdl-button--colored mdl-js-button mdl-button--primary mdl-js-ripple-effect">
            Start playing
          </button>
        </a>
      </div>
    </div>
  </div>
  <!-- Featured Card -->
  <div class="featured-demo mdl-grid">
    <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
    <div class="demo-card featured-demo mdl-card mdl-shadow--4dp feature-card mdl-cell mdl-cell--6-col">
      <a href="demos/performance_rnn">
        <div class="mdl-card__title" id="perf-rnn">
          <h1 class="mdl-card__title-text">Performance RNN</h1>
        </div>
      </a>
      <div class="mdl-card__supporting-text">Enjoy a real-time piano performance by a neural network</div>
      <div class="mdl-card__actions mdl-card--border">
        <a href="demos/performance_rnn">
          <button class="mdl-button mdl-button--raised mdl-button--colored mdl-js-button mdl-button--primary mdl-js-ripple-effect">
            Demo
          </button>
        </a>
        <a href="https://github.com/pair-code/deeplearnjs/tree/master/demos/performance_rnn">
          <button class="mdl-button mdl-button--raised mdl-button--colored mdl-js-button mdl-button--primary mdl-js-ripple-effect">
            Code
          </button>
        </a>
      </div>
    </div>
    <div class="demo-card featured-demo mdl-card mdl-shadow--4dp feature-card mdl-cell mdl-cell--6-col">
      <a href="https://teachablemachine.withgoogle.com/">
        <div class="mdl-card__title" id="teachable-machine">
          <h1 class="mdl-card__title-text">Teachable Machine</h1>
        </div>
      </a>
      <div class="mdl-card__supporting-text">Explore machine learning, no coding required!</div>
      <div class="mdl-card__actions mdl-card--border">
        <a href="https://teachablemachine.withgoogle.com/">
          <button class="mdl-button mdl-button--raised mdl-button--colored mdl-js-button mdl-button--primary mdl-js-ripple-effect">
            Demo
          </button>
        </a>
        <a href="https://github.com/googlecreativelab/teachable-machine">
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
      <a href="demos/model-builder/">
        <div class="mdl-card__title" id="model-builder">
          <span class="mdl-card__title-text">Model Builder</span>
        </div>
      </a>
      <div class="mdl-card__supporting-text">Build a neural network in your browser, without code!</div>
      <div class="mdl-card__actions mdl-card--border">
        <a href="demos/model-builder/">
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
    <div class="demo-card mdl-card mdl-shadow--2dp square-card mdl-cell mdl-cell--3-col mdl-cell--8-col-tablet mdl-cell--4-col-phone">
      <a href="demos/imagenet/">
        <div class="mdl-card__title" id="webcam">
          <span class="mdl-card__title-text">Webcam Imagenet</span>
        </div>
      </a>
      <div class="mdl-card__supporting-text">Squeezenet running in the browser</div>
      <div class="mdl-card__actions mdl-card--border">
        <a href="demos/imagenet/">
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
      <a href="demos/nn-art/">
        <div class="mdl-card__title" id="nnart">
          <span class="mdl-card__title-text">NNArt</span>
        </div>
      </a>
      <div class="mdl-card__supporting-text">Play with an animating CPPN</div>
      <div class="mdl-card__actions mdl-card--border">
        <a href="demos/nn-art/">
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
      <a href="demos/benchmarks/">
        <div class="mdl-card__title" id="benchmarks">
          <span class="mdl-card__title-text">Benchmarks</span>
        </div>
      </a>
      <div class="mdl-card__supporting-text">Test the library's performance</div>
      <div class="mdl-card__actions mdl-card--border">
        <a href="demos/benchmarks/">
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


<div class="mdl-grid">
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
  <div class="mdl-cell mdl-cell--10-col">
    {% capture my_include %}{% include README.md %}{% endcapture %}
    {{ my_include | markdownify }}
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
      <span class="deeplearn-shine">deeplearn.js</span> was originally developed by
      <a id="author1"></a>, <a id="author2"></a>, and
      <a href="https://twitter.com/c_nich">Charles Nicholson</a>.
    </p>
    <script>
      function daniel(elem) {
        elem.href = 'https://twitter.com/dsmilkov';
        elem.innerText = 'Daniel Smilkov';
      }
      function nikhil(elem) {
        elem.href = 'https://twitter.com/nsthorat';
        elem.innerText = 'Nikhil Thorat';
      }

      var author1 = document.getElementById('author1');
      var author2 = document.getElementById('author2');
      if (Math.random() > .5) {
        daniel(author1);
        nikhil(author2);
      } else {
        nikhil(author1);
        daniel(author2);
      }
    </script>
  </div>
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
  <div class= "ack mdl-cell mdl-cell--4-col mdl-cell--8-col-tablet mdl-cell-4-col-phone">
    <p class='intro-body mdl-typography--body-1'>
      We would like to acknowledge Chi Zeng, David Farhi, Mahima Pushkarna,
      Lauren Hannah-Murphy, Minsuk (Brian) Kahng, James Wexler, Martin Wattenberg,
      Fernanda Viégas, Greg Corrado, Jeff Dean for their tremendous help, and the
      Google Brain team for providing support for the project.
    </p>
  </div>
  <div class="mdl-layout-spacer mdl-cell--hide-tablet mdl-cell--hide-phone"></div>
</div>
<script src="bundle.js"></script>
