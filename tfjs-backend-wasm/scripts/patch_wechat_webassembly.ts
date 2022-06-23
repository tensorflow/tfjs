import {walk} from 'estree-walker';
import {FunctionDeclaration, BaseNode} from 'estree';
import {Plugin} from 'rollup';

const createInstantiateWasmFuncString = `
function createInstantiateWasmFunc(path) {
  return function (imports, callback) {
    WebAssembly.instantiate(path, imports).then(function (output) {
      callback(output.instance, output.module);
    });
    return {};
  };
}
`;

type NodeType = BaseNode | FunctionDeclaration & {start: number, end: number};
/**
 * WebAssembly has changed to WXWebAssembly after WeChat 8.0
 * 0. not simd or thread support.
 * 1. only load local wasm file is allowed.
 * 2. WebAssembly.validate not working so env registerFlag doesn't work
 * @see https://developers.weixin.qq.com/community/develop/doc/000e2c019f8a003d5dfbb54c251c00?jumpto=comment&commentid=000eac66934960576d0cb1a7256c
 */
export function patchWechatWebAssembly(): Plugin {
  return {
    name: 'patchWechatWebAssembly',
    transform(code, file) {
      // remove node imports
      if (
        file.endsWith('tfjs-backend-wasm-threaded-simd.worker.js') ||
        file.endsWith('tfjs-backend-wasm-threaded-simd.js')
      ) {
        code = code.replace(`require("worker_threads")`, 'null');
        code = code.replace(`require("perf_hooks")`, 'null');
      }

      // it is not a nice way, but WebAssembly.validate not working and SIMD is not support in WXWebAssembly
      // tf.env().set('WASM_HAS_SIMD_SUPPORT', false) will be done in tfjs-wechat or application code
      // so does the WASM_HAS_MULTITHREAD_SUPPORT
      if (file.endsWith('backend_wasm.ts')) {
        const ast = (this as {parse: (code: string) => BaseNode}).parse(code);
        walk(ast, {
          enter(node: NodeType) {
            if (
              node.type === 'FunctionDeclaration' &&
              'id' in node &&
              node.id.name === 'createInstantiateWasmFunc'
            ) {
              code = code.replace(
                code.slice(node.start, node.end),
                createInstantiateWasmFuncString,
              );
            }
          },
        });
      }

      code = code.replace(/WebAssembly\./g, `WXWebAssembly.`);
      code = code.replace(/typeof WebAssembly/g, `typeof WXWebAssembly`);
      return { code, map: null };
    },
  };
}
