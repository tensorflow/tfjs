require('../model_config.js');
const {write} = require('./app.js');
const {getTabId} = require('./index.js');
const browser_list = require('./browser_list.json');
const backends = ['cpu', 'wasm', 'webgl'];
const allBrowsers = new Set();
const allBenchmarkConfigs = [];

// Creates list of all testable backend and model combinations
for (const backend in backends) {
  for (const benchmark in benchmarks) {
    if (benchmark === 'custom' || benchmark.includes('USE')) continue;
    allBenchmarkConfigs.push(
        {'benchmark': {'model': benchmark, 'backend': backends[backend]}});
  }
}

// Creates list of all browser-device combinations in imported browser_list.json
for (const pairing in browser_list) {
  const browser = browser_list[pairing];
  allBrowsers[getTabId(browser)] = {
    'base': 'BrowserStack',
    'browser': browser.browser,
    'browser_version': browser.browser_version,
    'os': browser.os,
    'os_version': browser.os_version,
    'device': browser.device
  };
}

// Writes object including list of backend-model configs and all browsers
write(
    './all_configs.json',
    {'benchmarkConfigs': allBenchmarkConfigs, 'browsers': allBrowsers});
console.log('File created at ./all_configs.json');
