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

import {FusedConv2D, FusedConv2DAttrs, FusedConv2DInputs, KernelConfig, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {addConfig} from './Add';

import {conv2DConfig} from './Conv2D';

export const fusedConv2DConfig: KernelConfig = {
  kernelName: FusedConv2D,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {x, filter, bias} = inputs as FusedConv2DInputs;
    const {strides, pad, dataFormat, dilations, dimRoundingMode} =
        attrs as {} as FusedConv2DAttrs;
    const cpuBackend = backend as MathBackendCPU;

    const result = conv2DConfig.kernelFunc({
      inputs: {x, filter},
      backend,
      attrs: {strides, pad, dataFormat, dilations, dimRoundingMode}
    }) as TensorInfo;

    if (bias) {
      const resultWithBias =
          addConfig.kernelFunc({inputs: {a: result, b: bias}, backend}) as
          TensorInfo;

      cpuBackend.disposeData(result.dataId);

      return resultWithBias;
    }
    // if (activation) {
    //   result =
    //       mapActivation(this, result, activation, preluActivationWeights) as
    //       Tensor4D;
    // }
    return result;
  }
};
