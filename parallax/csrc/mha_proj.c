/*
 * Copyright (c) Alibaba Cloud Computing.
 *
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
 */

#include <parallax/csrc/mha_proj.h>
#include <parallax/native/mha_proj.h>
#include <parallax/tests/test_c_mha_proj.h>

void fused_mha_proj(const float* qkv, const float* weights, const int B, const int T, const int H_q, const int H_kv, const int D) {
    // Implement Fused MHA projection helper
    // ...
}
