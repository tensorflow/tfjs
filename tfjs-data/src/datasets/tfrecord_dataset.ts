import { TensorContainer } from '@tensorflow/tfjs-core';
import { Dataset } from '../dataset';
import { LazyIterator } from '../iterators/lazy_iterator';

// TODO: To be discussed here(TensorContainer)
export class TFRecordDataset extends Dataset<TensorContainer> {
  constructor(protected readonly input: any) {
    super();
  }

  async iterator(): Promise<LazyIterator<TensorContainer>> {
    return await this.input.iterator();
  }
}
