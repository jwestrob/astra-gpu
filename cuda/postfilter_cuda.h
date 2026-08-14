#ifndef PLAN7_GPU_POSTFILTER_CUDA_H
#define PLAN7_GPU_POSTFILTER_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "ssv_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

enum plan7_postfilter_abi {
  PLAN7_POSTFILTER_RECORD_VERSION = 1,
  PLAN7_POSTFILTER_RECORD_SIZE = 16
};

/*
 * Version 1 deliberately preserves plan7_bias_result as its 12-byte prefix.
 * A finite vfsc or +INFINITY is an exact external Viterbi score. A NaN vfsc
 * means that the caller must run the CPU Viterbi filter. -INFINITY is never a
 * valid encoding. Bias-rejected finite MSV rows still carry an external vfsc,
 * so changing --nobias does not require recomputing their Viterbi DP.
 * MSV/SSV overflow rows use CPU_REQUIRED: a negative F2 threshold can make
 * even P=0 proceed to Viterbi, so +INFINITY is not a threshold-independent
 * substitute for the exact CPU Viterbi result.
 * A DEFINITE_REJECT row with NaN filtersc/vfsc is an exact raw-F1 reject and
 * may only be skipped when the consumer is bound to the identical F1 value
 * used to generate this CSR batch. It must never enter FilterScores directly.
 */
typedef struct plan7_postfilter_result {
  uint32_t sequence_index;
  float filtersc;
  int16_t msv_numerator;
  uint8_t msv_status;
  uint8_t action;
  float vfsc;
} plan7_postfilter_result;

typedef struct plan7_viterbi_database plan7_viterbi_database;

/* Profiles are retained by the Python owner for this object's lifetime. */
int plan7_viterbi_database_create(const uintptr_t *profile_pointers,
                                  size_t profile_count,
                                  plan7_viterbi_database **database,
                                  char *error,
                                  size_t error_size);

int plan7_viterbi_database_destroy(plan7_viterbi_database **database,
                                   char *error,
                                   size_t error_size);

size_t plan7_viterbi_database_profile_count(
  const plan7_viterbi_database *database);

/* Check exact source identity, live snapshot, and packed SSV row affinity. */
int plan7_viterbi_database_matches_ssv(
  const plan7_viterbi_database *database,
  const uint8_t *packed_scores,
  size_t packed_score_count,
  const uintptr_t *source_profile_pointers,
  const plan7_ssv_f1_profile *profiles,
  size_t profile_count,
  char *error,
  size_t error_size);

/* Internal resident-buffer entry point used by plan7_ssv_sequence_batch. */
int plan7_postfilter_candidates_device(
  const plan7_viterbi_database *database,
  const uint8_t *device_residues,
  const uint64_t *device_sequence_offsets,
  const uint64_t *host_sequence_lengths,
  size_t sequence_count,
  const float *device_null_scores,
  const uint8_t *device_compact_scores,
  const plan7_ssv_f1_profile *device_f1_profiles,
  const uint8_t *device_tjb,
  const float *device_length_logp,
  const float *device_length_log1mp,
  const plan7_bias_profile *device_bias_profiles,
  const plan7_bias_candidate *device_candidates,
  const plan7_bias_candidate *host_candidates,
  plan7_bias_ssv_input *device_msv_inputs,
  size_t candidate_count,
  plan7_postfilter_result *host_results,
  char *error,
  size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
