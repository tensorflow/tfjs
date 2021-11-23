// This file is not pre-processed or downleveled to es5, so it must be es5
// compatible.

// Track and remove listeners previously added by emscripten. See pre.js for the
// other half of the logic.

var listenersAdded;
if (beforeListeners) {
  listenersAdded = {
    uncaughtException: process.listeners('uncaughtException').filter(
        function(listener) {
          return !beforeListeners.uncaughtException.indexOf(listener) > -1;
        }
    ),
    unhandledRejection: process.listeners('unhandledRejection').filter(
        function(listener) {
          return !beforeListeners.unhandledRejection.indexOf(listener) > -1;
        }
    ),
  };
}

var actualModule;
if (typeof WasmBackendModule !== 'undefined') {
  actualModule = WasmBackendModule;
} else if (typeof WasmBackendModuleThreadedSimd !== 'undefined') {
  actualModule = WasmBackendModuleThreadedSimd;
} else {
  throw new Error('Could not find wasm module in post.js');
}

if (listenersAdded) {
  // Patch the wasm module's dispose method to also unregister listeners.
  var tmpDispose = actualModule['_dispose'];
  actualModule['_dispose'] = function() {
    tmpDispose();
    listenersAdded.uncaughtException.forEach(function(listener) {
      process.removeListener('uncaughtException', listener);
    });
    listenersAdded.unhandledRejection.forEach(function(listener) {
      process.removeListener('unhandledRejection', listener);
    });
  }
}
