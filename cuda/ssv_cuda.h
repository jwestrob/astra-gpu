#ifndef PLAN7_GPU_SSV_CUDA_H
#define PLAN7_GPU_SSV_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum plan7_ssv_status {
  PLAN7_SSV_OK = 0,
  PLAN7_SSV_ERANGE = 16,
  PLAN7_SSV_ENORESULT = 19,
  PLAN7_SSV_EMPTY = 255
};

enum plan7_f1_action {
  PLAN7_F1_CPU_REQUIRED = 0,
  PLAN7_F1_DEFINITE_REJECT = 1
};

typedef struct {
  uint8_t xE;
  uint8_t status;
  uint8_t tjb;
  uint8_t reserved;
  int16_t numerator;
} plan7_ssv_result;

typedef struct plan7_ssv_sequence_batch plan7_ssv_sequence_batch;

int plan7_cuda_device_count(char *error, size_t error_size);
int plan7_tjb_for_length(float scale, uint64_t length);
int plan7_ssv_f1_decision(uint8_t status,
                          int16_t numerator,
                          uint64_t length,
                          float scale,
                          float m_mu,
                          float m_lambda,
                          double f1,
                          double *ret_p);

int plan7_ssv_sequence_batch_create(const uint8_t *residues,
                                    size_t residue_count,
                                    const uint64_t *offsets,
                                    size_t offset_count,
                                    int alphabet_size,
                                    plan7_ssv_sequence_batch **batch,
                                    char *error,
                                    size_t error_size);

int plan7_ssv_sequence_batch_destroy(plan7_ssv_sequence_batch **batch,
                                     char *error,
                                     size_t error_size);

int plan7_ssv_sequence_batch_filter(plan7_ssv_sequence_batch *batch,
                                    const uint8_t *striped_scores,
                                    size_t striped_score_count,
                                    int score_stride,
                                    int model_length,
                                    int alphabet_size,
                                    uint8_t tbm,
                                    uint8_t tec,
                                    uint8_t base,
                                    uint8_t bias,
                                    float scale,
                                    plan7_ssv_result *results,
                                    size_t result_count,
                                    char *error,
                                    size_t error_size);

int plan7_ssv_filter_cuda(const uint8_t *striped_scores,
                          size_t striped_score_count,
                          int score_stride,
                          int model_length,
                          int alphabet_size,
                          const uint8_t *residues,
                          size_t residue_count,
                          const uint64_t *offsets,
                          size_t offset_count,
                          size_t sequence_count,
                          uint8_t tbm,
                          uint8_t tec,
                          uint8_t base,
                          uint8_t bias,
                          float scale,
                          plan7_ssv_result *results,
                          size_t result_count,
                          char *error,
                          size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
