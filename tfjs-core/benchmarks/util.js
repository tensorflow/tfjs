function printTime(elapsed) {
  return elapsed.toFixed(1) + ' ms';
}

function printMemory(bytes) {
  if (bytes < 1024) {
    return bytes + ' B';
  } else if (bytes < 1024 * 1024) {
    return (bytes / 1024).toFixed(2) + ' KB';
  } else {
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }
}

function sleep(timeMs) {
  return new Promise(resolve => setTimeout(resolve, timeMs));
}

function queryTimerIsEnabled() {
  return _tfengine.ENV.getNumber(
             'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0;
}
