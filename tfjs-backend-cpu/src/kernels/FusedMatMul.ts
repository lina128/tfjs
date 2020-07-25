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

import {_FusedMatMul, _FusedMatMulAttrs, _FusedMatMulInputs, KernelConfig, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {addConfig} from './Add';
import {batchMatMulConfig} from './BatchMatMul';

export const fusedMatMulConfig: KernelConfig = {
  kernelName: _FusedMatMul,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {a, b, bias} = inputs as _FusedMatMulInputs;
    const {transposeA, transposeB, activation} =
        attrs as {} as _FusedMatMulAttrs;
    const cpuBackend = backend as MathBackendCPU;

    const result =
        batchMatMulConfig.kernelFunc(
            {inputs: {a, b}, backend, attrs: {transposeA, transposeB}}) as
        TensorInfo;

    if (bias) {
      const resultWithBias =
          addConfig.kernelFunc({inputs: {result, bias}, backend}) as TensorInfo;

      cpuBackend.disposeData(result.dataId);

      return resultWithBias;
    }

    if (activation) {
      // result = cpuBackend.mapActivation(
      //     this, result, activation, preluActivationWeights);
    }

    return result;
  }
};
