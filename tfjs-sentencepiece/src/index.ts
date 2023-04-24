import sentencePieceModuleFactory from './sentencepiece';

export async function prepare() {
  console.log('hello!!!');
  return await sentencePieceModuleFactory({});
}
