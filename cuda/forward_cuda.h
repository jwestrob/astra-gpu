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

typedef struct plan7_forward_database plan7_forward_database;
typedef struct plan7_ssv_sequence_batch plan7_ssv_sequence_batch;
typedef struct plan7_forward_output plan7_forward_output;

/* Profiles are retained by the Python owner for this object's lifetime. */
int plan7_forward_database_create(const uintptr_t *profile_pointers,
                                  size_t profile_count,
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

#ifdef __cplusplus
}
#endif

#endif
