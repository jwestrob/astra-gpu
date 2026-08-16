#ifndef PLAN7_GPU_FORWARD_CUDA_H
#define PLAN7_GPU_FORWARD_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum plan7_forward_abi {
  PLAN7_FORWARD_RECORD_VERSION = 1,
  PLAN7_FORWARD_RECORD_SIZE = 12,
  PLAN7_FORWARD_MAX_GATHERED_BYTES = 384 * 1024 * 1024
};

enum plan7_forward_action {
  PLAN7_FORWARD_CPU_REQUIRED = 0,
  PLAN7_FORWARD_DEFINITE_REJECT = 1,
  PLAN7_FORWARD_DEFINITE_PASS = 2
};

enum plan7_forward_status {
  PLAN7_FORWARD_OK = 0,
  PLAN7_FORWARD_ERANGE = 16,
  PLAN7_FORWARD_ENORESULT = 19,
  PLAN7_FORWARD_EMPTY = 255
};

/* One record is returned for every input F2 survivor, in input order.
 * Special-state rows are returned only for DEFINITE_PASS records. */
typedef struct plan7_forward_result {
  uint32_t sequence_index;
  float fwdsc;
  uint8_t status;
  uint8_t action;
  uint16_t reserved;
} plan7_forward_result;

typedef struct plan7_forward_statistics {
  uint64_t generation_f3_bits;
  uint64_t candidate_count;
  uint64_t survivor_count;
  uint64_t work_cells;
  uint64_t dp_workspace_bytes;
  uint64_t xmx_workspace_bytes;
  uint64_t gather_workspace_bytes;
  uint64_t gathered_xmx_bytes;
  uint64_t output_byte_limit;
  uint64_t output_cap_fallback_count;
  float kernel_milliseconds;
  float classification_milliseconds;
  float gather_milliseconds;
  float download_milliseconds;
  float total_milliseconds;
} plan7_forward_statistics;

/* Sealed identity of the exact resident profile database, sequence batch,
 * ordered F3-pass rows, and gathered Forward special-state bits. Later
 * stages must validate all fields before doing work. */
typedef struct plan7_forward_provenance {
  uint64_t database_generation;
  uint64_t batch_generation;
  uint64_t row_hash;
  uint64_t special_hash;
  uint64_t continuation_hash;
  uint64_t pass_count;
  uint64_t special_count;
  uint64_t generation_f3_bits;
  uint64_t integrity_tag;
} plan7_forward_provenance;

typedef struct plan7_forward_database plan7_forward_database;
typedef struct plan7_ssv_sequence_batch plan7_ssv_sequence_batch;
typedef struct plan7_forward_output plan7_forward_output;
typedef struct plan7_forward_workspace plan7_forward_workspace;

/* Pointer-free immutable host representation used by ProfileSelection. */
typedef struct plan7_forward_snapshot_profile {
  uint64_t emission_offset;
  uint64_t transition_offset;
  uint32_t q;
  uint32_t model_length;
  float e_move;
  float e_loop;
  float f_tau;
  float f_lambda;
  float nj;
  int32_t mode;
} plan7_forward_snapshot_profile;

/* Read-only internal view shared with later CUDA pipeline stages. Public
 * callers should treat every pointer as opaque device storage. */
typedef struct plan7_forward_device_profile {
  uint64_t emission_offset;
  uint64_t transition_offset;
  uint32_t q;
  uint32_t model_length;
  float e_move;
  float e_loop;
} plan7_forward_device_profile;

typedef struct plan7_forward_device_view {
  uint64_t generation_id;
  int32_t device_ordinal;
  int32_t alphabet_size;
  size_t profile_count;
  const plan7_forward_device_profile *profiles;
  const float *emissions;
  const float *transitions;
} plan7_forward_device_view;

enum plan7_forward_workspace_capacity {
  PLAN7_FORWARD_CAPACITY_CANDIDATE_PROFILES = 0,
  PLAN7_FORWARD_CAPACITY_CANDIDATE_SEQUENCES = 1,
  PLAN7_FORWARD_CAPACITY_LENGTH_TRANSITIONS = 2,
  PLAN7_FORWARD_CAPACITY_DP_OFFSETS = 3,
  PLAN7_FORWARD_CAPACITY_X_OFFSETS = 4,
  PLAN7_FORWARD_CAPACITY_DP = 5,
  PLAN7_FORWARD_CAPACITY_XMX = 6,
  PLAN7_FORWARD_CAPACITY_RESULTS = 7,
  PLAN7_FORWARD_CAPACITY_SURVIVOR_CANDIDATES = 8,
  PLAN7_FORWARD_CAPACITY_SURVIVOR_OFFSETS = 9,
  PLAN7_FORWARD_CAPACITY_GATHERED = 10,
  PLAN7_FORWARD_CAPACITY_COUNT = 11
};

typedef struct plan7_forward_workspace_statistics {
  uint64_t device_bytes;
  uint64_t dp_capacity_bytes;
  uint64_t xmx_capacity_bytes;
  uint64_t gather_capacity_bytes;
  uint64_t growth_count;
  uint64_t event_create_count;
  uint64_t run_count;
  uint64_t capacity_bytes[PLAN7_FORWARD_CAPACITY_COUNT];
} plan7_forward_workspace_statistics;

/* Profiles are retained by the Python owner for this object's lifetime. */
int plan7_forward_database_create(const uintptr_t *profile_pointers,
                                  size_t profile_count,
                                  plan7_forward_database **database,
                                  char *error,
                                  size_t error_size);

/* Stage one trusted immutable ProfileSelection snapshot on the current
 * device. All host arrays are consumed synchronously and need not outlive the
 * call. identity_tokens are opaque values and are never dereferenced. */
int plan7_forward_database_create_snapshot(
  int alphabet_size,
  const plan7_forward_snapshot_profile *profiles,
  size_t profile_count,
  const float *emissions,
  size_t emission_count,
  const float *transitions,
  size_t transition_count,
  const uintptr_t *identity_tokens,
  plan7_forward_database **database,
  char *error,
  size_t error_size);

int plan7_forward_database_destroy(plan7_forward_database **database,
                                   char *error,
                                   size_t error_size);

size_t plan7_forward_database_profile_count(
  const plan7_forward_database *database);

uint64_t plan7_forward_database_device_bytes(
  const plan7_forward_database *database);

float plan7_forward_database_pack_milliseconds(
  const plan7_forward_database *database);

float plan7_forward_database_upload_milliseconds(
  const plan7_forward_database *database);

int plan7_forward_database_get_device_view(
  const plan7_forward_database *database,
  plan7_forward_device_view *view,
  char *error,
  size_t error_size);

int plan7_forward_database_get_profile_snapshot(
  const plan7_forward_database *database,
  size_t profile_index,
  plan7_forward_snapshot_profile *profile,
  char *error,
  size_t error_size);

int plan7_forward_workspace_create(plan7_forward_workspace **workspace,
                                   char *error,
                                   size_t error_size);

int plan7_forward_workspace_destroy(plan7_forward_workspace **workspace,
                                    char *error,
                                    size_t error_size);

int plan7_forward_workspace_get_statistics(
  const plan7_forward_workspace *workspace,
  plan7_forward_workspace_statistics *statistics,
  char *error,
  size_t error_size);

/* candidate_offsets is a profile-major CSR indptr with profile_count+1
 * entries. special_offsets has candidate_count+1 float offsets. Rejected and
 * CPU_REQUIRED records have empty special-state spans. */
int plan7_forward_run(
  const plan7_forward_database *database,
  const plan7_ssv_sequence_batch *batch,
  const uintptr_t *source_profile_pointers,
  size_t profile_count,
  const uint64_t *candidate_offsets,
  const uint32_t *candidate_indices,
  const float *filter_scores,
  size_t candidate_count,
  double f3,
  uint64_t gathered_byte_budget,
  plan7_forward_output **output,
  char *error,
  size_t error_size);

/* Calls sharing a workspace must be serialized. */
int plan7_forward_run_with_workspace(
  plan7_forward_workspace *workspace,
  const plan7_forward_database *database,
  const plan7_ssv_sequence_batch *batch,
  const uintptr_t *source_profile_pointers,
  size_t profile_count,
  const uint64_t *candidate_offsets,
  const uint32_t *candidate_indices,
  const float *filter_scores,
  size_t candidate_count,
  double f3,
  uint64_t gathered_byte_budget,
  plan7_forward_output **output,
  char *error,
  size_t error_size);

/* Internal SequenceBatch entry point. Calls that share a batch must be
 * serialized, including calls to batch destruction or workspace statistics. */
int plan7_forward_run_batch_workspace(
  const plan7_forward_database *database,
  plan7_ssv_sequence_batch *batch,
  const uintptr_t *source_profile_pointers,
  size_t profile_count,
  const uint64_t *candidate_offsets,
  const uint32_t *candidate_indices,
  const float *filter_scores,
  size_t candidate_count,
  double f3,
  uint64_t gathered_byte_budget,
  plan7_forward_output **output,
  char *error,
  size_t error_size);

int plan7_forward_output_destroy(plan7_forward_output **output,
                                 char *error,
                                 size_t error_size);

size_t plan7_forward_output_result_count(const plan7_forward_output *output);
const plan7_forward_result *plan7_forward_output_results(
  const plan7_forward_output *output);
const uint64_t *plan7_forward_output_special_offsets(
  const plan7_forward_output *output);
size_t plan7_forward_output_special_count(const plan7_forward_output *output);
const float *plan7_forward_output_specials(
  const plan7_forward_output *output);
const plan7_forward_statistics *plan7_forward_output_statistics(
  const plan7_forward_output *output);
const plan7_forward_provenance *plan7_forward_output_provenance(
  const plan7_forward_output *output);

/* Validates the complete opaque token, including continuation_hash and the
 * generation threshold. The integrity tag detects accidental mixing and
 * mutation inside a trusted process; it is not a cryptographic MAC. */
int plan7_forward_database_validate_provenance(
  const plan7_forward_database *database,
  const plan7_forward_provenance *provenance);

#ifdef __cplusplus
}
#endif

#endif
