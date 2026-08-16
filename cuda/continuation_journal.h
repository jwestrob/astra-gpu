#ifndef PLAN7_GPU_CONTINUATION_JOURNAL_H
#define PLAN7_GPU_CONTINUATION_JOURNAL_H

#include <stddef.h>
#include <stdint.h>

#include "domain_rescore_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/* This is an in-process, one-shot transport between plan7_gpu._native and
 * plan7_gpu._pipeline.  Every address is represented as an offset from the
 * beginning of one allocation; the capsule that owns the allocation is the
 * only Python-visible object.  The integrity tag detects accidental damage
 * or object mixing; it is not a cryptographic boundary against deliberate
 * ctypes use or calls into package-private underscore APIs. */
enum plan7_continuation_journal_abi {
  PLAN7_CONTINUATION_JOURNAL_VERSION = 2,
  PLAN7_CONTINUATION_JOURNAL_MAGIC = 0x504a4e4c,
  PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE = 32
};

enum plan7_continuation_compact_route {
  PLAN7_CONTINUATION_COMPACT_NONE = 0,
  PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED = 1,
  PLAN7_CONTINUATION_COMPACT_DEVICE = 2
};

#define PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME \
  "plan7_gpu._native._continuation_journal_v2"
#define PLAN7_CONTINUATION_JOURNAL_CONSUMED_NAME \
  "plan7_gpu._pipeline._consumed_journal_v2"

typedef struct plan7_continuation_journal_row {
  uint32_t profile_index;
  uint32_t sequence_index;
  float usc;
  float filtersc;
  float vfsc;
  float fwdsc;
  float backward_score;
  float nexpected;
  uint32_t uncertain_count;
  uint32_t region_count;
  uint32_t multidomain_count;
  uint8_t postfilter_status;
  uint8_t postfilter_action;
  uint8_t forward_status;
  uint8_t forward_action;
  uint8_t domain_status;
  uint8_t domain_route;
  uint8_t has_own_scales;
  uint8_t reserved;
  uint32_t compact_result_count;
  uint8_t compact_route;
  uint8_t reserved2[3];
  uint32_t reserved3;
} plan7_continuation_journal_row;

typedef struct plan7_continuation_journal {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint32_t row_size;
  uint32_t region_size;
  uint32_t compact_result_size;
  uint32_t compact_trace_step_size;
  uint32_t compact_null2_stride;
  uint64_t total_bytes;

  uint64_t session_id;
  uint64_t selection_id;
  uint64_t profile_count;
  uint64_t postfilter_count;
  uint64_t forward_count;
  uint64_t row_count;
  uint64_t special_count;
  uint64_t region_count;

  uint64_t compact_result_count;
  uint64_t compact_trace_offset_count;
  uint64_t compact_trace_count;
  uint64_t compact_null2_count;
  uint64_t generation_tail_fingerprint;
  uint64_t rescore_simple_row_count;
  uint64_t rescore_device_result_count;
  uint64_t rescore_cpu_required_count;
  uint64_t rescore_numeric_fallback_count;
  uint64_t rescore_cap_fallback_count;
  uint64_t rescore_global_cpu_fallback_count;
  uint64_t rescore_compact_output_byte_limit;
  uint64_t rescore_compact_output_bytes;

  uint64_t generation_f1_bits;
  uint64_t generation_f2_bits;
  uint64_t generation_f3_bits;
  uint32_t rt1_bits;
  uint32_t rt2_bits;
  uint32_t rt3_bits;
  uint32_t guard_band_bits;
  uint8_t generation_bias_filter;
  uint8_t generation_compact_domains;
  uint8_t compact_global_fallback;
  uint8_t reserved0[5];
  uint8_t sequence_content_fingerprint[32];

  uint64_t postfilter_offsets_offset;
  uint64_t postfilter_records_offset;
  uint64_t forward_offsets_offset;
  uint64_t forward_records_offset;
  uint64_t forward_special_offsets_offset;
  uint64_t profile_offsets_offset;
  uint64_t identity_tokens_offset;
  uint64_t profile_fingerprints_offset;
  uint64_t rows_offset;
  uint64_t special_offsets_offset;
  uint64_t specials_offset;
  uint64_t region_offsets_offset;
  uint64_t regions_offset;
  uint64_t compact_row_offsets_offset;
  uint64_t compact_results_offset;
  uint64_t compact_trace_offsets_offset;
  uint64_t compact_traces_offset;
  uint64_t compact_null2_offset;

  plan7_forward_provenance forward;
  plan7_backward_domain_provenance backward;
  plan7_domain_rescore_provenance rescore;
  uint64_t integrity_tag;
} plan7_continuation_journal;

#ifdef __cplusplus
static_assert(sizeof(plan7_continuation_journal_row) == 64,
              "continuation journal row ABI changed");
static_assert(offsetof(plan7_continuation_journal, integrity_tag) == 776,
              "continuation journal integrity offset changed");
static_assert(sizeof(plan7_continuation_journal) == 784,
              "continuation journal header ABI changed");
#else
_Static_assert(sizeof(plan7_continuation_journal_row) == 64,
               "continuation journal row ABI changed");
_Static_assert(offsetof(plan7_continuation_journal, integrity_tag) == 776,
               "continuation journal integrity offset changed");
_Static_assert(sizeof(plan7_continuation_journal) == 784,
               "continuation journal header ABI changed");
#endif

static inline uint64_t plan7_continuation_journal_hash_bytes(
    uint64_t hash, const void *data, size_t size) {
  const uint8_t *bytes = (const uint8_t *)data;
  for (size_t i = 0; i < size; ++i) {
    hash ^= (uint64_t)bytes[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static inline uint64_t plan7_continuation_journal_hash_u32(
    uint64_t hash, uint32_t value) {
  for (unsigned shift = 0; shift != 32; shift += 8) {
    hash ^= (uint64_t)((uint8_t)(value >> shift));
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static inline uint64_t plan7_continuation_journal_hash_u64(
    uint64_t hash, uint64_t value) {
  for (unsigned shift = 0; shift != 64; shift += 8) {
    hash ^= (uint64_t)((uint8_t)(value >> shift));
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

/* Recompute the three canonical hashes produced by domain_rescore_cuda.cu.
 * Keeping this verifier in the transport header lets the CPU consumer
 * authenticate a copied journal without linking the PyHMMER companion
 * extension against CUDA. */
static inline int plan7_continuation_journal_rescore_hashes(
    const plan7_domain_rescore_result *results, uint64_t result_count,
    const uint64_t *trace_offsets,
    const plan7_domain_rescore_trace_step *traces, uint64_t trace_count,
    const float *null2, uint64_t null2_count,
    uint64_t *result_hash_out, uint64_t *trace_hash_out,
    uint64_t *null2_hash_out) {
  union { float value; uint32_t bits; } encoded;
  uint64_t result_hash = UINT64_C(1469598103934665603);
  uint64_t trace_hash = UINT64_C(1469598103934665603);
  uint64_t null2_hash = UINT64_C(1469598103934665603);
  uint64_t index;

  if (trace_offsets == NULL || result_hash_out == NULL ||
      trace_hash_out == NULL || null2_hash_out == NULL ||
      (result_count != 0 && results == NULL) ||
      (trace_count != 0 && traces == NULL) ||
      (null2_count != 0 && null2 == NULL) || trace_offsets[0] != 0 ||
      trace_offsets[result_count] != trace_count ||
      result_count > UINT64_MAX / PLAN7_DOMAIN_RESCORE_NULL2_COUNT ||
      null2_count != result_count * PLAN7_DOMAIN_RESCORE_NULL2_COUNT)
    return 0;

  result_hash = plan7_continuation_journal_hash_u64(
      result_hash, UINT64_C(0x44524553));
  trace_hash = plan7_continuation_journal_hash_u64(
      trace_hash, UINT64_C(0x54524345));
  null2_hash = plan7_continuation_journal_hash_u64(
      null2_hash, UINT64_C(0x4e554c32));
  for (index = 0; index < result_count; ++index) {
    const plan7_domain_rescore_result *result = results + index;
    const uint64_t begin = trace_offsets[index];
    const uint64_t end = trace_offsets[index + 1];
    uint64_t step;
    uint64_t residue;
    if (begin > end || end > trace_count) return 0;
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->row_index);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->profile_index);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->sequence_index);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->envelope_begin);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->envelope_end);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->alignment_begin);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->alignment_end);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->model_begin);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->model_end);
#define PLAN7_CONTINUATION_HASH_FLOAT(field) do {                             \
      encoded.value = result->field;                                         \
      result_hash = plan7_continuation_journal_hash_u32(                     \
          result_hash, encoded.bits);                                        \
    } while (0)
    PLAN7_CONTINUATION_HASH_FLOAT(forward_score);
    PLAN7_CONTINUATION_HASH_FLOAT(backward_score);
    PLAN7_CONTINUATION_HASH_FLOAT(oa_score);
    PLAN7_CONTINUATION_HASH_FLOAT(domain_correction);
    PLAN7_CONTINUATION_HASH_FLOAT(score_consistency);
#undef PLAN7_CONTINUATION_HASH_FLOAT
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, (uint32_t)result->status |
                         ((uint32_t)result->action << 8) |
                         ((uint32_t)result->has_own_scales << 16) |
                         ((uint32_t)result->reserved << 24));
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->reserved2);

    trace_hash = plan7_continuation_journal_hash_u64(
        trace_hash, end - begin);
    for (step = begin; step < end; ++step) {
      encoded.value = traces[step].posterior;
      trace_hash = plan7_continuation_journal_hash_u32(
          trace_hash, traces[step].sequence_position);
      trace_hash = plan7_continuation_journal_hash_u32(
          trace_hash, traces[step].model_position);
      trace_hash = plan7_continuation_journal_hash_u32(
          trace_hash, encoded.bits);
      trace_hash = plan7_continuation_journal_hash_u32(
          trace_hash, traces[step].state);
    }
    for (residue = 0; residue < PLAN7_DOMAIN_RESCORE_NULL2_COUNT; ++residue) {
      encoded.value = null2[
          index * PLAN7_DOMAIN_RESCORE_NULL2_COUNT + residue];
      null2_hash = plan7_continuation_journal_hash_u32(
          null2_hash, encoded.bits);
    }
  }
  *result_hash_out = plan7_continuation_journal_hash_u64(
      result_hash, PLAN7_DOMAIN_RESCORE_RECORD_VERSION);
  *trace_hash_out = plan7_continuation_journal_hash_u64(
      trace_hash, trace_count);
  *null2_hash_out = plan7_continuation_journal_hash_u64(
      null2_hash, null2_count);
  return 1;
}

static inline uint64_t plan7_continuation_journal_integrity(
    const plan7_continuation_journal *journal) {
  if (journal == NULL || journal->total_bytes < sizeof(*journal)) return 0;
  uint64_t hash = UINT64_C(1469598103934665603);
  hash = plan7_continuation_journal_hash_bytes(
      hash, journal, offsetof(plan7_continuation_journal, integrity_tag));
  return plan7_continuation_journal_hash_bytes(
      hash, (const uint8_t *)journal + sizeof(*journal),
      (size_t)(journal->total_bytes - sizeof(*journal)));
}

#ifdef __cplusplus
}
#endif

#endif
