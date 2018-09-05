import {renderConfusionMatrix} from '../render/confusion_matrix';
import {getDrawArea} from '../render/render_utils';
import {renderTable} from '../render/table';
import {ConfusionMatrixData, Drawable} from '../types';

export function perClassAccuracy(
    container: Drawable, classAccuracy: {accuracy: number[], count: number[]},
    classLabels?: string[]) {
  const drawArea = getDrawArea(container);

  const headers = [
    'Class',
    'Accuracy',
    '# Samples',
  ];
  const values: Array<Array<(string | number)>> = [];

  for (let i = 0; i < classAccuracy.accuracy.length; i++) {
    const label = classLabels ? classLabels[i] : i.toString();
    const acc = classAccuracy.accuracy[i];
    const count = classAccuracy.count[i];
    values.push([label, acc, count]);
  }

  renderTable({headers, values}, drawArea);
}

export function confusionMatrix(
    container: Drawable, confusionMatrix: number[][], classLabels?: string[]) {
  const drawArea = getDrawArea(container);

  const confusionMatrixData: ConfusionMatrixData = {
    values: confusionMatrix,
    labels: classLabels,
  };

  renderConfusionMatrix(confusionMatrixData, drawArea, {
    height: 450,
  });
}
