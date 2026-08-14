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
typedef struct plan7_postfilter_workspace plan7_postfilter_workspace;
typedef struct plan7_profile_session plan7_profile_session;
typedef struct plan7_profile_selection plan7_profile_selection;

/* Immutable, pointer-free host data for one ordered profile selection.  Every
 * pointer remains valid until the selection is destroyed.  The identity
 * tokens are opaque values: they are compared but never dereferenced. */
typedef struct plan7_profile_selection_view {
  uint64_t session_id;
  uint64_t selection_id;
  size_t profile_count;
  const uint8_t *packed_scores;
  size_t packed_score_count;
  const plan7_ssv_profile *profiles;
  const float *m_mu;
  const float *m_lambda;
  const plan7_bias_profile *bias_templates;
  const uintptr_t *identity_tokens;
  uint64_t host_bytes;
} plan7_profile_selection_view;

typedef struct plan7_profile_session_statistics {
  uint64_t session_id;
  uint64_t profile_count;
  uint64_t worker_count;
  uint64_t selection_count;
  uint64_t parallel_run_count;
  /* Immutable payload bytes, excluding allocator slack and worker stacks. */
  uint64_t host_bytes;
  uint64_t ssv_score_bytes;
  uint64_t bias_profile_bytes;
  uint64_t viterbi_descriptor_bytes;
  uint64_t viterbi_emission_bytes;
  uint64_t viterbi_transition_bytes;
  uint64_t viterbi_exact_rbv_bytes;
} plan7_profile_session_statistics;

enum plan7_postfilter_workspace_capacity {
  PLAN7_POSTFILTER_CAPACITY_STATES = 0,
  PLAN7_POSTFILTER_CAPACITY_BIAS_INPUTS = 1,
  PLAN7_POSTFILTER_CAPACITY_BIAS_RESULTS = 2,
  PLAN7_POSTFILTER_CAPACITY_VITERBI_RESULTS = 3,
  PLAN7_POSTFILTER_CAPACITY_LENGTH_TRANSITIONS = 4,
  PLAN7_POSTFILTER_CAPACITY_MSV_OFFSETS = 5,
  PLAN7_POSTFILTER_CAPACITY_VITERBI_OFFSETS = 6,
  PLAN7_POSTFILTER_CAPACITY_DP = 7,
  PLAN7_POSTFILTER_CAPACITY_RESULTS = 8,
  PLAN7_POSTFILTER_CAPACITY_COUNT = 9
};

typedef struct plan7_postfilter_workspace_statistics {
  uint64_t device_bytes;
  uint64_t dp_capacity_bytes;
  uint64_t growth_count;
  uint64_t run_count;
  uint64_t capacity_bytes[PLAN7_POSTFILTER_CAPACITY_COUNT];
} plan7_postfilter_workspace_statistics;

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

/* Snapshot private optimized-profile arrays into immutable host storage.
 * The caller must prevent every source profile from being mutated until this
 * function returns.  No CUDA context or device allocation is used here. */
int plan7_profile_session_create(const uintptr_t *profile_pointers,
                                 size_t profile_count,
                                 const float *background,
                                 size_t background_count,
                                 plan7_profile_session **session,
                                 char *error,
                                 size_t error_size);

int plan7_profile_session_destroy(plan7_profile_session **session,
                                  char *error,
                                  size_t error_size);

int plan7_profile_session_get_statistics(
  const plan7_profile_session *session,
  plan7_profile_session_statistics *statistics,
  char *error,
  size_t error_size);

/* Selection order is semantically significant.  Indexes must be unique and
 * in range.  Calls on a session, including destroy, must be serialized. */
int plan7_profile_session_select(plan7_profile_session *session,
                                 const size_t *profile_indices,
                                 size_t profile_count,
                                 plan7_profile_selection **selection,
                                 char *error,
                                 size_t error_size);

int plan7_profile_selection_destroy(plan7_profile_selection **selection,
                                    char *error,
                                    size_t error_size);

/* Destroy must be serialized against view and staging calls on the same
 * selection.  A selection remains valid after its source session closes. */
int plan7_profile_selection_get_view(
  const plan7_profile_selection *selection,
  plan7_profile_selection_view *view,
  char *error,
  size_t error_size);

/* Upload only this selection to the current device.  The resulting database
 * contains sealed host provenance and never reads a live P7_OPROFILE. */
int plan7_profile_selection_stage_viterbi(
  const plan7_profile_selection *selection,
  plan7_viterbi_database **database,
  char *error,
  size_t error_size);

/* Check exact source identity, packed SSV row affinity, and (for legacy
 * databases) the live optimized-profile snapshot. */
int plan7_viterbi_database_matches_ssv(
  const plan7_viterbi_database *database,
  const uint8_t *packed_scores,
  size_t packed_score_count,
  const uintptr_t *source_profile_pointers,
  const plan7_ssv_f1_profile *profiles,
  size_t profile_count,
  char *error,
  size_t error_size);

int plan7_postfilter_workspace_create(plan7_postfilter_workspace **workspace,
                                      char *error,
                                      size_t error_size);

int plan7_postfilter_workspace_destroy(plan7_postfilter_workspace **workspace,
                                       char *error,
                                       size_t error_size);

int plan7_postfilter_workspace_get_statistics(
  const plan7_postfilter_workspace *workspace,
  plan7_postfilter_workspace_statistics *statistics,
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

/* Serialized resident-workspace variant used by plan7_ssv_sequence_batch.
 * The workspace and database must belong to the current CUDA device. */
int plan7_postfilter_candidates_device_with_workspace(
  plan7_postfilter_workspace *workspace,
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
