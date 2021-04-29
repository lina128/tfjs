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

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';

describeWithFlags('sparseFillEmptyRows', ALL_ENVS, () => {
  it('fill number', async () => {
    const sparseTensor = {
      ind: tf.tensor2d(
          [[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]], [6, 2], 'int32'),
      val: [0, 10, 13, 14, 32, 33],
      shape: [5, 6],
    };
    const result = tf.sparse.sparseFillEmptyRows(
        sparseTensor.ind, sparseTensor.val, sparseTensor.shape, -1);
    expectArraysClose(
        await result.outputIndices.data(),
        [[0, 0], [1, 0], [1, 3], [1, 4], [2, 0], [3, 2], [3, 3], [4, 0]]);
    expectArraysClose(
        await result.outputValues.data(), [0, 10, 13, 14, -1, 32, 33, -1]);
    expectArraysClose(await result.emptyRowIndicator.data(), [0, 0, 1, 0, 1]);
    expectArraysClose(await result.reverseIndexMap.data(), [0, 1, 2, 3, 5, 6]);
  });

  it('fill float', async () => {
    const sparseTensor = {
      ind: tf.tensor2d(
          [[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]], [6, 2], 'float32'),
      val: [0.0, 10.0, 13.0, 14.0, 32.0, 33.0],
      shape: [5, 6],
    };
    const result = tf.sparse.sparseFillEmptyRows(
        sparseTensor.ind, sparseTensor.val, sparseTensor.shape, -1);
    expectArraysClose(
        await result.outputIndices.data(),
        [[0, 0], [1, 0], [1, 3], [1, 4], [2, 0], [3, 2], [3, 3], [4, 0]]);
    expectArraysClose(
        await result.outputValues.data(), [0, 10, 13, 14, -1, 32, 33, -1]);
    expectArraysClose(await result.emptyRowIndicator.data(), [0, 0, 1, 0, 1]);
    expectArraysClose(await result.reverseIndexMap.data(), [0, 1, 2, 3, 5, 6]);
  });

  it('no empty rows', async () => {
    const sparseTensor = {
      ind: tf.tensor2d([[0, 0], [1, 0], [1, 3], [1, 4]], [4, 2], 'int32'),
      val: [0, 10, 13, 14],
      shape: [2, 6],
    };
    const result = tf.sparse.sparseFillEmptyRows(
        sparseTensor.ind, sparseTensor.val, sparseTensor.shape, -1);
    expectArraysClose(
        await result.outputIndices.data(), [[0, 0], [1, 0], [1, 3], [1, 4]]);
    expectArraysClose(await result.outputValues.data(), [0, 10, 13, 14]);
    expectArraysClose(await result.emptyRowIndicator.data(), [0, 0]);
    expectArraysClose(await result.reverseIndexMap.data(), [0, 1, 2, 3]);
  });

  it('no empty rows and unordered', async () => {
    const sparseTensor = {
      ind: tf.tensor2d([[1, 2], [1, 3], [0, 1], [0, 3]], [4, 2], 'float32'),
      val: [1, 3, 2, 4],
      shape: [2, 5],
    };
    const result = tf.sparse.sparseFillEmptyRows(
        sparseTensor.ind, sparseTensor.val, sparseTensor.shape, -1);
    expectArraysClose(
        await result.outputIndices.data(), [[0, 1], [0, 3], [1, 2], [1, 3]]);
    expectArraysClose(await result.outputValues.data(), [2, 4, 1, 3]);
    expectArraysClose(await result.emptyRowIndicator.data(), [0, 0]);
    expectArraysClose(await result.reverseIndexMap.data(), [2, 3, 0, 1]);
  });

  it('no rows', async () => {
    const sparseTensor = {
      ind: tf.tensor2d([], [0, 2], 'int32'),
      val: [] as number[],
      shape: [0, 5],
    };
    const result = tf.sparse.sparseFillEmptyRows(
        sparseTensor.ind, sparseTensor.val, sparseTensor.shape, -1);
    expectArraysClose(await result.outputIndices.data(), []);
    expectArraysClose(await result.outputValues.data(), []);
    expectArraysClose(await result.emptyRowIndicator.data(), []);
    expectArraysClose(await result.reverseIndexMap.data(), []);
  });

  it('throw error if dense rows is empty and indices is not', async () => {
    const sparseTensor = {
      ind: tf.tensor2d([[0, 0]], [1, 2], 'int32'),
      val: [1],
      shape: [0, 5],
    };
    expect(
        () => tf.sparse.sparseFillEmptyRows(
            sparseTensor.ind, sparseTensor.val, sparseTensor.shape, -1))
        .toThrowError(/indices\.shape\[0\] = 1/);
  });

  it('throw error if negative row', async () => {
    const sparseTensor = {
      ind: tf.tensor2d([[-1, 0]], [1, 2], 'int32'),
      val: [1],
      shape: [5, 5],
    };
    expect(
        () => tf.sparse.sparseFillEmptyRows(
            sparseTensor.ind, sparseTensor.val, sparseTensor.shape, -1))
        .toThrowError('indices(0, 0) is invalid: -1 < 0');
  });

  it('throw error if row exceeds number of dense rows', async () => {
    const sparseTensor = {
      ind: tf.tensor2d([[5, 0]], [1, 2], 'int32'),
      val: [1],
      shape: [5, 5],
    };
    expect(
        () => tf.sparse.sparseFillEmptyRows(
            sparseTensor.ind, sparseTensor.val, sparseTensor.shape, -1))
        .toThrowError('indices(0, 0) is invalid: 5 >= 5');
  });
});