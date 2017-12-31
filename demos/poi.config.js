const path = require('path');

module.exports = options => ({
  webpack(config) {
    config.output.publicPath = '';
    return config;
  },
  presets: [require('poi-preset-typescript')({
    loaderOptions: {
      configFile: options.mode === 'development' ?
          path.join('..', 'tsconfig-dev.json') :
          path.join('..', 'tsconfig.json')
    }
  })],
});
