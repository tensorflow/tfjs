/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {util} from '@tensorflow/tfjs-core';

/**
 * The StringNGramsOp class creates ngrams from ragged string data.
 * The constructor contains all attributes related to the operation such as
 * padding widths and strings, and the compute function can be used to
 * compute the ngrams for different ragged tensor inputs.
 */
class StringNGramsOp {
  private separator: Uint8Array;
  private nGramWidths: number[];
  private padWidth: number;
  private leftPad: Uint8Array;
  private rightPad: Uint8Array;
  private preserveShort: boolean;

  constructor(
      separator: string, nGramWidths: number[], leftPad: string,
      rightPad: string, padWidth: number, preserveShortSequences: boolean) {
    this.separator = util.encodeString(separator);
    this.nGramWidths = nGramWidths;
    this.leftPad = util.encodeString(leftPad);
    this.rightPad = util.encodeString(rightPad);
    this.padWidth = padWidth;
    this.preserveShort = preserveShortSequences;
  }

  private getPadWidth(nGramWidth: number) {
    // Ngrams can be padded with either a fixed pad width or a dynamic pad
    // width depending on the 'padWidth' arg, but in no case should the padding
    // ever be wider than 'nGramWidth' - 1.
    return Math.min(
        this.padWidth < 0 ? nGramWidth - 1 : this.padWidth, nGramWidth - 1);
  }

  private getNumNGrams(length: number, nGramWidth: number) {
    const padWidth = this.getPadWidth(nGramWidth);
    return Math.max(0, ((length + 2 * padWidth) - nGramWidth) + 1);
  }

  private createNGrams(
      data: Uint8Array[], splitIndex: number, output: Uint8Array[],
      outputStartIndex: number, numNGrams: number, nGramWidth: number) {
    for (let nGramIndex = 0; nGramIndex < numNGrams; ++nGramIndex) {
      const padWidth = this.getPadWidth(nGramWidth);
      const leftPadding = Math.max(0, padWidth - nGramIndex);
      const rightPadding =
          Math.max(0, padWidth - (numNGrams - (nGramIndex + 1)));
      const numTokens = nGramWidth - (leftPadding + rightPadding);
      const dataStartIndex =
          splitIndex + (leftPadding > 0 ? 0 : nGramIndex - padWidth);

      // Calculate the total expected size of the nGram so we can reserve the
      // correct amount of space in the string.
      let nGramSize = 0;
      // Size of the left padding.
      nGramSize += leftPadding * this.leftPad.length;
      // Size of the tokens.
      for (let n = 0; n < numTokens; ++n) {
        nGramSize += data[dataStartIndex + n].length;
      }
      // Size of the right padding.
      nGramSize += rightPadding * this.rightPad.length;
      // Size of the separators.
      const numSeparators = leftPadding + rightPadding + numTokens - 1;
      nGramSize += numSeparators * this.separator.length;

      // Build the nGram.
      output[outputStartIndex + nGramIndex] = new Uint8Array(nGramSize);
      const nGram = output[outputStartIndex + nGramIndex];

      let nextNGramIndex = 0;
      const appendToNGram = (str: Uint8Array) =>
          str.forEach((value) => nGram[nextNGramIndex++] = value);

      for (let n = 0; n < leftPadding; ++n) {
        appendToNGram(this.leftPad);
        appendToNGram(this.separator);
      }
      // Only output first numTokens - 1 pairs of data and separator
      for (let n = 0; n < numTokens - 1; ++n) {
        appendToNGram(data[dataStartIndex + n]);
        appendToNGram(this.separator);
      }
      // Handle case when there are no tokens or no right padding as these
      // can result in consecutive separators.
      if (numTokens > 0) {
        // If we have tokens, then output last and then pair each separator
        // with the right padding that follows, to ensure nGram ends either with
        // the token or with the right pad.
        appendToNGram(data[dataStartIndex + numTokens - 1]);
        for (let n = 0; n < rightPadding; ++n) {
          appendToNGram(this.separator);
          appendToNGram(this.rightPad);
        }
      } else {
        // If we don't have tokens, then the last item inserted into the nGram
        // has been the separator from the left padding loop above. Hence,
        // output right pad and separator and make sure to finish with a
        // padding, not a separator.
        for (let n = 0; n < rightPadding - 1; ++n) {
          appendToNGram(this.rightPad);
          appendToNGram(this.separator);
        }
        appendToNGram(this.rightPad);
      }
    }
  }

  // Data and splits together form the definition of the ragged tensor,
  // where data is 1 dimensional and contains the values of the tensor
  // and splits denotes the indices at which each row starts.
  public compute(data: Uint8Array[], splits: Int32Array):
      [Uint8Array[], Int32Array] {
    // Validate that the splits are valid indices into data, only if there are
    // splits specified.
    const inputDataSize = data.length;
    const splitsSize = splits.length;
    if (splitsSize > 0) {
      let prevSplit = splits[0];
      if (prevSplit !== 0) {
        throw new Error(`First split value must be 0, got ${prevSplit}`);
      }
      for (let i = 1; i < splitsSize; ++i) {
        let validSplits = splits[i] >= prevSplit;
        validSplits = validSplits && (splits[i] <= inputDataSize);
        if (!validSplits) {
          throw new Error(`Invalid split value ${splits[i]}, must be in [${
              prevSplit}, ${inputDataSize}]`);
        }
        prevSplit = splits[i];
      }
      if (prevSplit !== inputDataSize) {
        throw new Error(`Last split value must be data size. Expected ${
            inputDataSize}, got ${prevSplit}`);
      }
    }

    const numBatchItems = splitsSize - 1;
    const nGramsSplits = util.getArrayFromDType('int32', splitsSize);
    // If there is no data or size, return an empty ragged tensor.
    if (inputDataSize === 0 || splitsSize === 0) {
      const empty: Uint8Array[] = new Array(inputDataSize);
      for (let i = 0; i <= numBatchItems; ++i) {
        nGramsSplits[i] = 0;
      }
      return [empty, nGramsSplits];
    }

    nGramsSplits[0] = 0;
    for (let i = 1; i <= numBatchItems; ++i) {
      const length = splits[i] - splits[i - 1];
      let numNGrams = 0;
      this.nGramWidths.forEach((nGramWidth) => {
        numNGrams += this.getNumNGrams(length, nGramWidth);
      });
      if (this.preserveShort && length > 0 && numNGrams === 0) {
        numNGrams = 1;
      }
      nGramsSplits[i] = nGramsSplits[i - 1] + numNGrams;
    }

    const nGrams: Uint8Array[] = new Array(nGramsSplits[numBatchItems]);

    for (let i = 0; i < numBatchItems; ++i) {
      const splitIndex = splits[i];
      let outputStartIdx = nGramsSplits[i];
      this.nGramWidths.forEach((nGramWidth) => {
        const length = splits[i + 1] - splits[i];
        const numNGrams = this.getNumNGrams(length, nGramWidth);
        this.createNGrams(
            data, splitIndex, nGrams, outputStartIdx, numNGrams, nGramWidth);
        outputStartIdx += numNGrams;
      });
      // If we're preserving short sequences, check to see if no sequence was
      // generated by comparing the current output start idx to the original
      // one (nGramSplitsdata). If no ngrams were generated, then they will
      // be equal (since we increment outputStartIdx by numNGrams every
      // time we create a set of ngrams.)
      if (this.preserveShort && outputStartIdx === nGramsSplits[i]) {
        const dataLength = splits[i + 1] - splits[i];
        // One legitimate reason to not have any ngrams when this.preserveShort
        // is true is if the sequence itself is empty. In that case, move on.
        if (dataLength === 0) {
          continue;
        }
        // We don't have to worry about dynamic padding sizes here: if padding
        // was dynamic, every sequence would have had sufficient padding to
        // generate at least one nGram.
        const nGramWidth = dataLength + 2 * this.padWidth;
        const numNGrams = 1;
        this.createNGrams(
            data, splitIndex, nGrams, outputStartIdx, numNGrams, nGramWidth);
      }
    }
    return [nGrams, nGramsSplits];
  }
}

export function stringNGramsImpl(
    data: Uint8Array[], dataSplits: Int32Array, separator: string,
    nGramWidths: number[], leftPad: string, rightPad: string, padWidth: number,
    preserveShortSequences: boolean): [Uint8Array[], Int32Array] {
  return new StringNGramsOp(
             separator, nGramWidths, leftPad, rightPad, padWidth,
             preserveShortSequences)
      .compute(data, dataSplits);
}
