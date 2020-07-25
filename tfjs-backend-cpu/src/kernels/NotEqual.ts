/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {BinaryInputs, KernelConfig, KernelFunc, NotEqual, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {broadcastedBinaryOp} from '../utils/binary_utils';

const notEqual_: KernelFunc = ({inputs, backend}) => {
  const {a, b} = inputs as BinaryInputs;
  const cpuBackend = backend as MathBackendCPU;

  assertNotComplex([a, b], 'notEqual');

  const aVals = cpuBackend.data.get(a.dataId).values as TypedArray;
  const bVals = cpuBackend.data.get(b.dataId).values as TypedArray;

  const {values, shape} = broadcastedBinaryOp(
      aVals, bVals, a.shape, b.shape, 'bool', (aVal, bVal) => {
        return (aVal !== bVal) ? 1 : 0;
      });

  const dataId = cpuBackend.write(values, shape, 'bool');

  return {dataId, shape, dtype: 'bool'};
};

export const notEqualConfig: KernelConfig = {
  kernelName: NotEqual,
  backendName: 'cpu',
  kernelFunc: notEqual_
};
