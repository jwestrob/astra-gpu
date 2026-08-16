#ifndef PLAN7_GPU_CONTINUATION_JOURNAL_H
#define PLAN7_GPU_CONTINUATION_JOURNAL_H

#include <stddef.h>
#include <stdint.h>

#include "backward_domain_cuda.h"

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
  PLAN7_CONTINUATION_JOURNAL_VERSION = 1,
  PLAN7_CONTINUATION_JOURNAL_MAGIC = 0x504a4e4c,
  PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE = 32
};

#define PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME \
  "plan7_gpu._native._continuation_journal_v1"
#define PLAN7_CONTINUATION_JOURNAL_CONSUMED_NAME \
  "plan7_gpu._pipeline._consumed_journal_v1"

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
} plan7_continuation_journal_row;

typedef struct plan7_continuation_journal {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint32_t row_size;
  uint32_t region_size;
  uint64_t total_bytes;

  uint64_t session_id;
  uint64_t selection_id;
  uint64_t profile_count;
  uint64_t postfilter_count;
  uint64_t forward_count;
  uint64_t row_count;
  uint64_t special_count;
  uint64_t region_count;

  uint64_t generation_f1_bits;
  uint64_t generation_f2_bits;
  uint64_t generation_f3_bits;
  uint32_t rt1_bits;
  uint32_t rt2_bits;
  uint32_t rt3_bits;
  uint32_t guard_band_bits;
  uint8_t generation_bias_filter;
  uint8_t reserved0[7];
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

  plan7_forward_provenance forward;
  plan7_backward_domain_provenance backward;
  uint64_t integrity_tag;
} plan7_continuation_journal;

#ifdef __cplusplus
static_assert(sizeof(plan7_continuation_journal_row) == 52,
              "continuation journal row ABI changed");
static_assert(offsetof(plan7_continuation_journal, integrity_tag) == 456,
              "continuation journal integrity offset changed");
static_assert(sizeof(plan7_continuation_journal) == 464,
              "continuation journal header ABI changed");
#else
_Static_assert(sizeof(plan7_continuation_journal_row) == 52,
               "continuation journal row ABI changed");
_Static_assert(offsetof(plan7_continuation_journal, integrity_tag) == 456,
               "continuation journal integrity offset changed");
_Static_assert(sizeof(plan7_continuation_journal) == 464,
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
