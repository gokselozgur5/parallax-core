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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BLOCK_SIZE 128
#define NUM_HEADS 4
#define SEQ_LENGTH 512

typedef struct {
    float *q;
    float *k;
    float *v;
} FlashAttentionInput;

typedef struct {
    float *o;
} FlashAttentionOutput;

void flash_attention_neon(FlashAttentionInput *input, FlashAttentionOutput *output) {
    int i, j, k;
    int num_blocks = SEQ_LENGTH / BLOCK_SIZE;

    for (i = 0; i < num_blocks; ++i) {
        for (j = 0; j < NUM_HEADS; ++j) {
            for (k = 0; k < BLOCK_SIZE; ++k) {
                float q_val = input->q[i * BLOCK_SIZE + j * BLOCK_SIZE + k];
                float k_val = input->k[i * BLOCK_SIZE + j * BLOCK_SIZE + k];
                float v_val = input->v[i * BLOCK_SIZE + j * BLOCK_SIZE + k];

                // Online softmax
                float sum_exp = 0.0f;
                for (int l = 0; l < BLOCK_SIZE; ++l) {
                    float exp_val = exp(q_val - k_val);
                    sum_exp += exp_val;
                }

                // Compute attention value
                float attn_val = q_val * k_val / sum_exp;

                // Store the result in output
                output->o[i * BLOCK_SIZE + j * BLOCK_SIZE + k] = attn_val;
            }
        }
    }
}

void test_flash_attention_neon() {
    float *q = (float *)malloc(SEQ_LENGTH * NUM_HEADS * BLOCK_SIZE * sizeof(float));
    float *k = (float *)malloc(SEQ_LENGTH * NUM_HEADS * BLOCK_SIZE * sizeof(float));
    float *v = (float *)malloc(SEQ_LENGTH * NUM_HEADS * BLOCK_SIZE * sizeof(float));
    float *o = (float *)malloc(SEQ_LENGTH * NUM_HEADS * BLOCK_SIZE * sizeof(float));

    // Initialize input data
    for (int i = 0; i < SEQ_LENGTH * NUM_HEADS * BLOCK_SIZE; ++i) {
        q[i] = rand() / RAND_MAX;
        k[i] = rand() / RAND_MAX;
        v[i] = rand() / RAND_MAX;
    }

    // Create input and output structures
    FlashAttentionInput input = {q, k, v};
    FlashAttentionOutput output = o;

    // Perform flash attention computation
    flash_attention_neon(&input, &output);

    // Verify the result against reference attention
    float *ref_o = (float *)malloc(SEQ_LENGTH * NUM_HEADS * BLOCK_SIZE * sizeof(float));
    for (int i = 0; i < SEQ_LENGTH * NUM_HEADS * BLOCK_SIZE; ++i) {
        ref_o[i] = q[i] * k[i] / sum_exp;
    }

    // Compare the results
    int pass = 1;
    for (int i = 0; i < SEQ_LENGTH * NUM_HEADS * BLOCK_SIZE; ++i) {
        if (fabs(output.o[i] - ref_o[i]) > 1e-6f) {
            printf("Test failed at index %d\n", i);
            pass = 0;
            break;
