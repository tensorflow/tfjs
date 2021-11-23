// This file is not pre-processed or downleveled to es5, so it must be es5
// compatible.

// Keep track of listeners that are added by emscripten so they can later be
// removed. See post.js for the other half of the logic.

var beforeListeners;
if (typeof process !== 'undefined' && process.listeners) {
  beforeListeners = {
    uncaughtException: process.listeners('uncaughtException'),
    unhandledRejection: process.listeners('unhandledRejection'),
  }
}
