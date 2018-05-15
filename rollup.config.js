const definition = require("./package.json");
import node from "rollup-plugin-node-resolve";
import typescript from "rollup-plugin-typescript2";
import commonjs from "rollup-plugin-commonjs";
import resolve from "rollup-plugin-node-resolve";

export default {
  input: "src/index.ts",
  plugins: [
    typescript(),
    node(),
    // Polyfill require() from dependencies.
    commonjs({
      include: 'node_modules/**',
      namedExports: {
        './node_modules/seedrandom/index.js': ['alea']
      },
    })
  ],
  output: {
    extend: true,
    banner: `// @tensorflow/tfjs Copyright ${(new Date).getFullYear()} Google`,
    file: "dist/bundle.js",
    format: "umd",
    name: "tf",
    globals: {
      'crypto': 'crypto'
    }
  },
  external: ['crypto'],
  onwarn: warning => {
    let {code} = warning;
    if (code === 'CIRCULAR_DEPENDENCY' ||
        code === 'CIRCULAR' ||
        code === 'THIS_IS_UNDEFINED') {
      return;
    }
    console.warn('WARNING: ', warning.toString());
  }
};
