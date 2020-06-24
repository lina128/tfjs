/**
 * @license
 * Copyright 2020 Google LLC All Rights Reserved.
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

import {AvgPool3D, AvgPool3DAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {avgPool3dBackprop} from '../ops/avg_pool_3d_backprop';
import {Tensor, Tensor5D} from '../tensor';

export const avgPool3DGradConfig: GradConfig = {
  kernelName: AvgPool3D,
  inputsToSave: ['x'],
  gradFunc: (dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const [x] = saved as [Tensor5D];
    const {filterSize, strides, dilations, pad, dimRoundingMode} =
        attrs as {} as AvgPool3DAttrs;

    const $dilations =
        dilations == null ? [1, 1, 1] as [number, number, number] : dilations;

    return {
      x: () => avgPool3dBackprop(
          dy as Tensor5D, x, filterSize, strides, $dilations, pad,
          dimRoundingMode)
    };
  }
};
