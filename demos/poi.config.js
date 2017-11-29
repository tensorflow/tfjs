module.exports = {
  webpack(config) {
    config.output.publicPath = '';
    return config;
  },
  presets: [
    require('poi-preset-typescript')(/* options */)
  ],
};
