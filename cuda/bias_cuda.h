#ifndef PLAN7_GPU_BIAS_CUDA_H
#define PLAN7_GPU_BIAS_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum plan7_bias_action {
  PLAN7_BIAS_CPU_REQUIRED = 0,
  PLAN7_BIAS_DEFINITE_REJECT = 1,
  PLAN7_BIAS_DEFINITE_PASS = 2
};

enum plan7_bias_cutoff_mode {
  PLAN7_BIAS_CUTOFF_INVALID = 0,
  PLAN7_BIAS_CUTOFF_SCORE = 1,
  PLAN7_BIAS_CUTOFF_ALWAYS_REJECT = 2,
  PLAN7_BIAS_CUTOFF_ALWAYS_PASS = 3
};

typedef struct {
  float t10;
  float t11;
  float scale;
  float cutoff_bit_score;
  int32_t cutoff_mode;
  uint32_t reserved;
  float pi0;
  float pi1;
  float t02;
  float t12;
  float eo[29][2];
} plan7_bias_profile;

typedef struct {
  uint32_t profile_index;
  uint32_t sequence_index;
} plan7_bias_candidate;

typedef struct {
  int16_t numerator;
  uint8_t status;
  uint8_t reserved;
} plan7_bias_ssv_input;

typedef struct {
  uint32_t sequence_index;
  float filtersc;
  int16_t ssv_numerator;
  uint8_t ssv_status;
  uint8_t action;
} plan7_bias_result;

int plan7_bias_pack_amino_profile(const float *background,
                                  const float *composition,
                                  int model_length,
                                  float scale,
                                  int cutoff_mode,
                                  float cutoff_bit_score,
                                  plan7_bias_profile *profile,
                                  char *error,
                                  size_t error_size);

int plan7_bias_length_terms(uint64_t length,
                            float *length_logp,
                            float *length_log1mp);

int plan7_bias_filter_score_host(const plan7_bias_profile *profile,
                                 const uint8_t *residues,
                                 uint64_t length,
                                 float *filtersc);

int plan7_bias_rebias_decision(uint8_t ssv_status,
                               int16_t ssv_numerator,
                               float scale,
                               float filtersc,
                               int cutoff_mode,
                               float cutoff_bit_score,
                               float *bit_score);

/* Returns 1 only when host binary32/binary64 operations use the attested
 * rounding and denormal modes. */
int plan7_bias_host_environment_attested(void);

/* Returns 1 only for the exhaustively attested host/device math target. */
int plan7_bias_environment_attested(char *reason, size_t reason_size);

/* Internal device-buffer entry point used by the resident SSV batch. */
int plan7_bias_filter_candidates_device(
  const uint8_t *device_residues,
  const uint64_t *device_offsets,
  const float *device_length_logp,
  const float *device_length_log1mp,
  const plan7_bias_profile *device_profiles,
  const plan7_bias_candidate *device_candidates,
  const plan7_bias_ssv_input *device_ssv_inputs,
  size_t candidate_count,
  plan7_bias_result *device_results,
  char *error,
  size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
