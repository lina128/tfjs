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
// We explicitly import the modular kernels so they get registered in the
// global registry when we compile the library. A modular build would replace
// the contents of this file and import only the kernels that are needed.
import {KernelConfig, registerKernel} from '@tensorflow/tfjs-core';

import {addConfig} from './kernels/Add';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {castConfig} from './kernels/Cast';
import {complexConfig} from './kernels/Complex';
import {conv2DConfig} from './kernels/Conv2D';
import {depthwiseConv2dNativeConfig} from './kernels/DepthwiseConv2dNative';
import {dilation2dConfig} from './kernels/Dilation2D';
import {dilation2dBackpropFilterConfig} from './kernels/Dilation2DBackpropFilter';
import {dilation2dBackpropInputConfig} from './kernels/Dilation2DBackpropInput';
import {divConfig} from './kernels/Div';
import {expConfig} from './kernels/Exp';
import {fusedConv2DConfig} from './kernels/FusedConv2d';
import {fusedDepthwisesConv2DConfig} from './kernels/FusedDepthwiseConv2D';
import {fusedMatMulConfig} from './kernels/FusedMatMul';
import {identityConfig} from './kernels/Identity';
import {imagConfig} from './kernels/Imag';
import {intConfig} from './kernels/Int';
import {maxConfig} from './kernels/Max';
import {maxPoolWithArgmaxConfig} from './kernels/MaxPoolWithArgmax';
import {multiplyConfig} from './kernels/Multiply';
import {negateConfig} from './kernels/Negate';
import {nonMaxSuppressionV4Config} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {notEqualConfig} from './kernels/NotEqual';
import {realConfig} from './kernels/Real';
import {reshapeConfig} from './kernels/Reshape';
import {rotateWithOffsetConfig} from './kernels/RotateWithOffset';
import {softmaxConfig} from './kernels/Softmax';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {subConfig} from './kernels/Sub';
import {sumConfig} from './kernels/Sum';
import {transposeConfig} from './kernels/Transpose';
import {zerosConfig} from './kernels/Zeros';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  addConfig,
  conv2DConfig,
  batchMatMulConfig,
  castConfig,
  complexConfig,
  dilation2dConfig,
  dilation2dBackpropInputConfig,
  dilation2dBackpropFilterConfig,
  divConfig,
  depthwiseConv2dNativeConfig,
  expConfig,
  identityConfig,
  intConfig,
  imagConfig,
  fusedConv2DConfig,
  fusedDepthwisesConv2DConfig,
  fusedMatMulConfig,
  maxPoolWithArgmaxConfig,
  maxConfig,
  multiplyConfig,
  negateConfig,
  nonMaxSuppressionV4Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  realConfig,
  reshapeConfig,
  rotateWithOffsetConfig,
  squareConfig,
  squaredDifferenceConfig,
  softmaxConfig,
  subConfig,
  sumConfig,
  transposeConfig,
  zerosConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
