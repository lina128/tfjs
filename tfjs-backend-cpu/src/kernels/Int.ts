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

import {Int, IntInputs, KernelConfig, KernelFunc, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

const int_: KernelFunc = ({inputs, backend}) => {
  const {x} = inputs as IntInputs;
  const cpuBackend = backend as MathBackendCPU;

  assertNotComplex(x, 'int');

  const xVals = cpuBackend.data.get(x.dataId).values as TypedArray;
  const outVals = new Int32Array(xVals.length);

  for (let i = 0; i < xVals.length; ++i) {
    outVals[i] = xVals[i];
  }

  const dataId = cpuBackend.write(outVals, x.shape, 'int32');

  return {dataId, shape: x.shape, dtype: 'int32'};
};

export const intConfig: KernelConfig = {
  kernelName: Int,
  backendName: 'cpu',
  kernelFunc: int_
};
