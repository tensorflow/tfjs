import {GraphNode, registerOp} from '@tensorflow/tfjs-converter';
import {Tensor, tensor, tensor1d, util} from '@tensorflow/tfjs-core';

import sentencePieceModuleFactory from './sentencepiece';
import {SentencePieceModule, Vector} from './sentencepiece';

type MultiDimsNumberArray = number|number[]|number[][]|number[][][]|
    number[][][][]|number[][][][][]|number[][][][][][];

export async function loadModule() {
  return await sentencePieceModuleFactory({});
}

const modulePromise: Promise<SentencePieceModule> = loadModule();

function vectorPush<T>(vec: Vector<T>, values: T[]): Vector<T> {
  for (const value of values) vec.push_back(value);
  return vec;
}

function vectorToArray<T>(vec: Vector<T>): T[] {
  const values: T[] = [];
  for (let i = 0; i < vec.size(); ++i) {
    values.push(vec.get(i));
  }
  return values;
}

/**
 * @param modelSerializedProto Serialized proto binary in string.
 * @returns The unique model key.
 */
export async function registerModel(modelSerializedProto: string):
    Promise<string> {
  return (await modulePromise).RegisterModelBase64(atob(modelSerializedProto));
}

/**
 * @param modelSerializedProtoBase64 Serialized proto binary in base64 string.
 * @returns The unique model key.
 */
export async function registerModelBase64(modelSerializedProtoBase64: string):
    Promise<string> {
  return (await modulePromise).RegisterModelBase64(modelSerializedProtoBase64);
}

export async function encodeString(
    modelKey: string, strings: string[]|Tensor, addBos = false, addEos = false,
    reverse = false) {
  const m = await modulePromise;
  const stringValues =
      (strings instanceof Tensor ? await strings.array() : strings) as
      unknown as string[];
  const stringsVec = vectorPush(new m.VectorString(), stringValues);
  const result = m.EncodeString(modelKey, stringsVec, addBos, addEos, reverse);

  const tokens = tensor1d(vectorToArray(result.valuesFlat));
  const splits = tensor1d(vectorToArray(result.splitsFlat));

  result.valuesFlat.delete();
  result.splitsFlat.delete();
  stringsVec.delete();

  return {tokens, splits};
}

export async function decodeString(
    modelKey: string, tokens: MultiDimsNumberArray|Tensor,
    splits: MultiDimsNumberArray|Tensor, addBos = false, addEos = false,
    reverse = false) {
  const m = await modulePromise;
  const tokenValues =
      util.flatten(tokens instanceof Tensor ? await tokens.array() : tokens);
  const splitValues =
      util.flatten(splits instanceof Tensor ? await splits.array() : splits);

  const tokensVec = vectorPush(new m.VectorInt(), tokenValues);
  const splitsVec = vectorPush(new m.VectorInt(), splitValues);

  const outputsVec =
      m.DecodeString(modelKey, tokensVec, splitsVec, addBos, addEos, reverse);
  const outputs = vectorToArray(outputsVec);

  tokensVec.delete();
  splitsVec.delete();
  outputsVec.delete();

  return tensor(outputs);
}

export async function registerOps() {
  registerOp('SentencepieceOp', async (node: GraphNode) => {
    const modelSerializedProto = node.attrs['model'] as string;
    const modelKey = await registerModel(modelSerializedProto);
    return [tensor(modelKey)];
  });

  registerOp('SentencepieceTokenizeOp', async (node: GraphNode) => {
    const modelKey = (await node.inputs[0].array()) as unknown as string;
    const strings = (await node.inputs[1].array()) as unknown as string[];
    const addBos = (await node.inputs[4].array()) as unknown as boolean;
    const addEos = (await node.inputs[5].array()) as unknown as boolean;
    const reverse = (await node.inputs[6].array()) as unknown as boolean;

    const {tokens, splits} =
        await encodeString(modelKey, strings, addBos, addEos, reverse);

    return [tokens, splits];
  });

  registerOp('SentencepieceDetokenizeOp', async (node: GraphNode) => {
    const modelKey = (await node.inputs[0].array()) as unknown as string;
    const tokens = await node.inputs[1].array();
    const splits = await node.inputs[2].array();
    const addBos = (await node.inputs[3].array()) as unknown as boolean;
    const addEos = (await node.inputs[4].array()) as unknown as boolean;
    const reverse = (await node.inputs[5].array()) as unknown as boolean;

    return [await decodeString(
        modelKey, tokens, splits, addBos, addEos, reverse)];
  });
}
