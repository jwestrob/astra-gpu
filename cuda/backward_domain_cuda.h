#ifndef PLAN7_GPU_BACKWARD_DOMAIN_CUDA_H
#define PLAN7_GPU_BACKWARD_DOMAIN_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "forward_cuda.h"
#include "ssv_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

enum plan7_backward_domain_abi {
  PLAN7_BACKWARD_DOMAIN_RECORD_VERSION = 2,
  PLAN7_BACKWARD_DOMAIN_RECORD_SIZE = 32,
  PLAN7_BACKWARD_DOMAIN_POSTERIOR_SIZE = 12,
  PLAN7_BACKWARD_DOMAIN_REGION_SIZE = 8,
  PLAN7_BACKWARD_DOMAIN_MAX_POSTERIOR_BYTES = 384 * 1024 * 1024,
  PLAN7_BACKWARD_DOMAIN_MAX_ROW_WORK_CELLS = 10000000,
  PLAN7_BACKWARD_DOMAIN_MAX_RUN_WORK_CELLS = 256 * 1024 * 1024
};

enum plan7_backward_domain_route {
  PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED = 0,
  PLAN7_BACKWARD_DOMAIN_NO_REGIONS = 1,
  PLAN7_BACKWARD_DOMAIN_SIMPLE = 2
};

enum plan7_backward_domain_status {
  PLAN7_BACKWARD_DOMAIN_OK = 0,
  PLAN7_BACKWARD_DOMAIN_ERANGE = 16,
  PLAN7_BACKWARD_DOMAIN_ENORESULT = 19,
  PLAN7_BACKWARD_DOMAIN_EMPTY = 255
};

/* The row record is pointer-free: profiles and sequences are identified by
 * index, and Forward parser state is an ordinary packed array. */
typedef struct plan7_backward_domain_candidate {
  uint32_t profile_index;
  uint32_t sequence_index;
} plan7_backward_domain_candidate;

typedef struct plan7_backward_domain_result {
  uint32_t profile_index;
  uint32_t sequence_index;
  float backward_score;
  float nexpected;
  uint32_t uncertain_count;
  uint32_t region_count;
  uint32_t multidomain_count;
  uint8_t status;
  uint8_t route;
  uint8_t has_own_scales;
  uint8_t reserved;
} plan7_backward_domain_result;

typedef struct plan7_domain_posterior {
  float btot;
  float etot;
  float mocc;
} plan7_domain_posterior;

typedef struct plan7_simple_region {
  uint32_t begin;
  uint32_t end;
} plan7_simple_region;

/* Seals the exact Forward generation, decision thresholds, ordered result
 * records, and compact interval journal consumed by the continuation seam. */
typedef struct plan7_backward_domain_provenance {
  plan7_forward_provenance forward;
  uint64_t threshold_hash;
  uint64_t result_hash;
  uint64_t region_hash;
  uint64_t candidate_count;
  uint64_t region_count;
} plan7_backward_domain_provenance;

typedef struct plan7_backward_domain_statistics {
  uint64_t candidate_count;
  uint64_t device_result_count;
  uint64_t cpu_required_count;
  uint64_t work_cells;
  uint64_t dp_workspace_bytes;
  uint64_t backward_special_workspace_bytes;
  uint64_t forward_special_workspace_bytes;
  uint64_t posterior_bytes;
  uint64_t simple_region_bytes;
  uint64_t output_byte_limit;
  uint64_t output_cap_fallback_count;
  uint64_t work_cap_fallback_count;
  uint64_t posterior_omitted_count;
  uint64_t own_scale_count;
  uint64_t threshold_uncertain_count;
  uint64_t no_region_count;
  uint64_t simple_count;
  uint64_t multidomain_fallback_count;
  float kernel_milliseconds;
  float upload_milliseconds;
  float download_milliseconds;
  float total_milliseconds;
} plan7_backward_domain_statistics;

typedef struct plan7_backward_domain_output plan7_backward_domain_output;

/* Low-level diagnostic seam. Calls must be serialized with all other uses and
 * destruction of database and batch. The implementation snapshots all raw
 * caller arrays before validation. Forward offsets contain float offsets and
 * must span exactly 6*(L+1) values for each row. Rows must be profile-major
 * and target indexes must be strictly increasing within a profile. A future
 * production wrapper must consume sealed CandidateBatch/Forward handles and
 * enforce the calibrated minimum guard; raw guard_band=0 is oracle-only. */
int plan7_backward_domain_run(
  const plan7_forward_database *database,
  const plan7_ssv_sequence_batch *batch,
  const plan7_backward_domain_candidate *candidates,
  size_t candidate_count,
  const plan7_forward_provenance *provenance,
  const uint64_t *forward_offsets,
  const float *forward_specials,
  size_t forward_special_count,
  float rt1,
  float rt2,
  float rt3,
  float guard_band,
  uint64_t posterior_byte_budget,
  plan7_backward_domain_output **output,
  char *error,
  size_t error_size);

/* Test-only recurrence seam for synthetic Forward scale fixtures. It never
 * validates provenance or seals output and forces every returned route to
 * CPU_REQUIRED, so its journal cannot be consumed by production code. */
int plan7_backward_domain_unsealed_test_run(
  const plan7_forward_database *database,
  const plan7_ssv_sequence_batch *batch,
  const plan7_backward_domain_candidate *candidates,
  size_t candidate_count,
  const uint64_t *forward_offsets,
  const float *forward_specials,
  size_t forward_special_count,
  float rt1,
  float rt2,
  float rt3,
  float guard_band,
  uint64_t posterior_byte_budget,
  plan7_backward_domain_output **output,
  char *error,
  size_t error_size);

int plan7_backward_domain_output_destroy(
  plan7_backward_domain_output **output,
  char *error,
  size_t error_size);

size_t plan7_backward_domain_output_result_count(
  const plan7_backward_domain_output *output);
const plan7_backward_domain_result *plan7_backward_domain_output_results(
  const plan7_backward_domain_output *output);
const uint64_t *plan7_backward_domain_output_posterior_offsets(
  const plan7_backward_domain_output *output);
size_t plan7_backward_domain_output_posterior_count(
  const plan7_backward_domain_output *output);
const plan7_domain_posterior *plan7_backward_domain_output_posteriors(
  const plan7_backward_domain_output *output);
const uint64_t *plan7_backward_domain_output_region_offsets(
  const plan7_backward_domain_output *output);
size_t plan7_backward_domain_output_region_count(
  const plan7_backward_domain_output *output);
const plan7_simple_region *plan7_backward_domain_output_regions(
  const plan7_backward_domain_output *output);
const plan7_backward_domain_provenance *
plan7_backward_domain_output_provenance(
  const plan7_backward_domain_output *output);
const plan7_backward_domain_statistics *
plan7_backward_domain_output_statistics(
  const plan7_backward_domain_output *output);

/* Pristine HMMER 3.4 SSE oracle for focused tests. The only pointer-valued
 * argument is deliberately confined to this test seam; production work uses
 * the pointer-free indexed ABI above. Input residues are 0-based digital
 * codes, while this helper constructs HMMER's sentinel-bearing ESL_DSQ. */
int plan7_backward_domain_cpu_oracle(
  uintptr_t source_profile_pointer,
  const uint8_t *residues,
  size_t residue_count,
  const float *forward_specials,
  size_t forward_special_count,
  float rt1,
  float rt2,
  float rt3,
  float guard_band,
  plan7_backward_domain_result *result,
  plan7_domain_posterior *posteriors,
  size_t posterior_count,
  plan7_simple_region *regions,
  size_t region_capacity,
  size_t *region_count,
  char *error,
  size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
