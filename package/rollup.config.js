import ascii from "rollup-plugin-ascii";
import node from "rollup-plugin-node-resolve";
import commonjs from "rollup-plugin-commonjs";

export default {
  input: "dist/index",
  plugins: [node(), commonjs(), ascii()],
  output: {
    extend: true,
    file: "dist/tf.js",
    format: "umd",
    name: "tf"
  }
};
