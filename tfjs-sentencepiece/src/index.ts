import {GraphNode, registerOp} from '@tensorflow/tfjs-converter';
import {tensor, tensor1d, util} from '@tensorflow/tfjs-core';

import sentencePieceModuleFactory from './sentencepiece';
import {SentencePieceModule, Vector} from './sentencepiece';

async function loadModule() {
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

export async function registerOps() {
  const m = await modulePromise;

  registerOp('SentencepieceOp', async (node: GraphNode) => {
    const modelSerializedProto = node.attrs['model'] as string;
    const modelKey = m.RegisterModel(modelSerializedProto);
    return [tensor(modelKey)];
  });

  registerOp('SentencepieceTokenizeOp', async (node: GraphNode) => {
    const modelKey = (await node.inputs[0].array()) as unknown as string;
    const strings = (await node.inputs[1].array()) as unknown as string[];
    const addBos = (await node.inputs[4].array()) as unknown as boolean;
    const addEos = (await node.inputs[5].array()) as unknown as boolean;
    const reverse = (await node.inputs[6].array()) as unknown as boolean;

    const stringsVec = vectorPush(new m.VectorString(), strings);
    const result =
        m.EncodeString(modelKey, stringsVec, addBos, addEos, reverse);

    const tokens = tensor1d(vectorToArray(result.valuesFlat));
    const splits = tensor1d(vectorToArray(result.splitsFlat));

    result.valuesFlat.delete();
    result.splitsFlat.delete();
    stringsVec.delete();

    return [tokens, splits];
  });

  registerOp('SentencepieceDetokenizeOp', async (node: GraphNode) => {
    const modelKey = (await node.inputs[0].array()) as unknown as string;
    const tokens = await node.inputs[1].array();
    const splits = await node.inputs[2].array();
    const addBos = (await node.inputs[3].array()) as unknown as boolean;
    const addEos = (await node.inputs[4].array()) as unknown as boolean;
    const reverse = (await node.inputs[5].array()) as unknown as boolean;

    const tokensVec = vectorPush(new m.VectorInt(), util.flatten(tokens));
    const splitsVec = vectorPush(new m.VectorInt(), util.flatten(splits));

    const outputsVec =
        m.DecodeString(modelKey, tokensVec, splitsVec, addBos, addEos, reverse);
    const ouptuts = vectorToArray(outputsVec);

    tokensVec.delete();
    splitsVec.delete();
    outputsVec.delete();

    return [tensor(ouptuts)];
  });
}
