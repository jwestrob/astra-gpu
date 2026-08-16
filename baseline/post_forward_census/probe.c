#define _GNU_SOURCE

#include <dlfcn.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "hmmer.h"

enum metric_id {
  METRIC_BACKWARD_PARSER,
  METRIC_DOMAIN_WORKFLOW,
  METRIC_DOMAIN_DECODING,
  METRIC_FORWARD,
  METRIC_BACKWARD,
  METRIC_STOCHASTIC_TRACE,
  METRIC_CLUSTER,
  METRIC_DECODING,
  METRIC_NULL2_EXPECTATION,
  METRIC_NULL2_TRACE,
  METRIC_OPTIMAL_ACCURACY,
  METRIC_OA_TRACE,
  METRIC_ALIDISPLAY,
  METRIC_TOPHITS_SORT,
  METRIC_TOPHITS_THRESHOLD,
  METRIC_COUNT
};

struct metric {
  const char *name;
  atomic_uint_fast64_t calls;
  atomic_uint_fast64_t elapsed_ns;
  atomic_uint_fast64_t ok;
  atomic_uint_fast64_t errors;
};

#define METRIC(name) { (name), ATOMIC_VAR_INIT(0), ATOMIC_VAR_INIT(0), \
                       ATOMIC_VAR_INIT(0), ATOMIC_VAR_INIT(0) }

static struct metric metrics[METRIC_COUNT] = {
  METRIC("p7_BackwardParser"),
  METRIC("p7_domaindef_ByPosteriorHeuristics"),
  METRIC("p7_DomainDecoding"),
  METRIC("p7_Forward"),
  METRIC("p7_Backward"),
  METRIC("p7_StochasticTrace"),
  METRIC("p7_spensemble_Cluster"),
  METRIC("p7_Decoding"),
  METRIC("p7_Null2_ByExpectation"),
  METRIC("p7_Null2_ByTrace"),
  METRIC("p7_OptimalAccuracy"),
  METRIC("p7_OATrace"),
  METRIC("p7_alidisplay_Create"),
  METRIC("p7_tophits_SortBySortkey"),
  METRIC("p7_tophits_Threshold"),
};

struct row {
  uint64_t serial;
  int64_t sequence_idx;
  char sequence_name[96];
  char profile_name[96];
  int L;
  int M;
  int status;
  uint64_t duration_ns;
  int backward_status;
  uint64_t backward_ns;
  uint64_t decoding_calls;
  uint64_t decoding_ns;
  uint32_t rng_seed;
  int do_reseeding;
  int nsamples;
  float nexpected;
  float rt1;
  float rt2;
  float rt3;
  int nregions;
  int replay_regions;
  int nclustered;
  int replay_clustered;
  int noverlaps;
  int nenvelopes;
  int ndom;
  uint64_t stochastic_calls;
  uint64_t stochastic_failures;
  uint64_t sampled_segments;
  uint64_t raw_clusters;
  int simple_regions;
  int final_cluster_envelopes;
  uint64_t region_residues;
  int region_max;
  uint64_t clustered_region_residues;
  int clustered_region_max;
  uint64_t envelope_residues;
  int envelope_max;
  uint64_t alignment_residues;
  int alignment_max;
  float min_rt1_margin;
  float min_start_rt2_margin;
  float min_end_rt2_margin;
  float min_rt3_margin;
  float min_b_delta;
  float min_e_delta;
  float min_mocc;
  float max_mocc;
  float terminal_be_delta;
  uint64_t nonfinite_posterior;
  int unclosed_region;
  uint64_t oxf_bytes;
  uint64_t oxb_bytes;
  uint64_t fwd_bytes;
  uint64_t bck_bytes;
  uint64_t ddef_bytes;
  uint64_t active_domain_bytes;
};

struct tls_state {
  const ESL_DSQ *backward_dsq;
  const P7_OPROFILE *backward_om;
  int backward_L;
  int backward_status;
  uint64_t backward_ns;
  int in_domain;
  uint64_t decoding_calls;
  uint64_t decoding_ns;
  uint64_t stochastic_calls;
  uint64_t stochastic_failures;
  uint64_t sampled_segments;
  uint64_t raw_clusters;
};

static _Thread_local struct tls_state tls;
static struct row *rows;
static size_t row_count;
static size_t row_capacity;
static pthread_mutex_t rows_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static atomic_uint_fast64_t next_serial = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t dropped_rows = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t clock_errors = ATOMIC_VAR_INIT(0);
static char output_path[4096];
static char target_library[4096] = "unknown";
static pid_t origin_pid;
static int enabled;
static void *hmmer_handle;
static void *easel_handle;

typedef int (*backward_parser_fn)(const ESL_DSQ *, int, const P7_OPROFILE *,
                                  const P7_OMX *, P7_OMX *, float *);
typedef int (*domain_workflow_fn)(const ESL_SQ *, const ESL_SQ *, P7_OPROFILE *,
                                  P7_OMX *, P7_OMX *, P7_OMX *, P7_OMX *,
                                  P7_DOMAINDEF *, P7_BG *, int, P7_BG *, float *,
                                  float *);
typedef int (*domain_decoding_fn)(const P7_OPROFILE *, const P7_OMX *,
                                  const P7_OMX *, P7_DOMAINDEF *);
typedef int (*forward_fn)(const ESL_DSQ *, int, const P7_OPROFILE *, P7_OMX *,
                          float *);
typedef int (*backward_fn)(const ESL_DSQ *, int, const P7_OPROFILE *,
                           const P7_OMX *, P7_OMX *, float *);
typedef int (*stochastic_fn)(ESL_RANDOMNESS *, const ESL_DSQ *, int,
                             const P7_OPROFILE *, const P7_OMX *, P7_TRACE *);
typedef int (*cluster_fn)(P7_SPENSEMBLE *, float, int, int, float, float, int *);
typedef int (*decoding_fn)(const P7_OPROFILE *, const P7_OMX *, P7_OMX *,
                           P7_OMX *);
typedef int (*null2_expectation_fn)(const P7_OPROFILE *, const P7_OMX *, float *);
typedef int (*null2_trace_fn)(const P7_OPROFILE *, const P7_TRACE *, int, int,
                              P7_OMX *, float *);
typedef int (*optimal_accuracy_fn)(const P7_OPROFILE *, const P7_OMX *, P7_OMX *,
                                   float *);
typedef int (*oa_trace_fn)(const P7_OPROFILE *, const P7_OMX *, const P7_OMX *,
                           P7_TRACE *);
typedef P7_ALIDISPLAY *(*alidisplay_fn)(const P7_TRACE *, int,
                                       const P7_OPROFILE *, const ESL_SQ *,
                                       const ESL_SQ *);
typedef int (*tophits_sort_fn)(P7_TOPHITS *);
typedef int (*tophits_threshold_fn)(P7_TOPHITS *, P7_PIPELINE *);
typedef uint32_t (*get_seed_fn)(const ESL_RANDOMNESS *);

static backward_parser_fn real_p7_BackwardParser;
static domain_workflow_fn real_p7_domaindef_ByPosteriorHeuristics;
static domain_decoding_fn real_p7_DomainDecoding;
static forward_fn real_p7_Forward;
static backward_fn real_p7_Backward;
static stochastic_fn real_p7_StochasticTrace;
static cluster_fn real_p7_spensemble_Cluster;
static decoding_fn real_p7_Decoding;
static null2_expectation_fn real_p7_Null2_ByExpectation;
static null2_trace_fn real_p7_Null2_ByTrace;
static optimal_accuracy_fn real_p7_OptimalAccuracy;
static oa_trace_fn real_p7_OATrace;
static alidisplay_fn real_p7_alidisplay_Create;
static tophits_sort_fn real_p7_tophits_SortBySortkey;
static tophits_threshold_fn real_p7_tophits_Threshold;
static get_seed_fn real_esl_randomness_GetSeed;

static void
fatal_symbol(const char *name)
{
  char message[256];
  ssize_t ignored = 0;
  int n = snprintf(message, sizeof(message),
                   "post_forward_census: cannot resolve %s\n", name);
  if (n > 0) ignored = write(STDERR_FILENO, message, (size_t) n);
  (void) ignored;
  _exit(127);
}

static void *
lookup_symbol(const char *name, int hmmer_only)
{
  void *symbol = NULL;
  if (hmmer_only && hmmer_handle != NULL) symbol = dlsym(hmmer_handle, name);
  if (symbol == NULL) symbol = dlsym(RTLD_NEXT, name);
  if (symbol == NULL) fatal_symbol(name);
  return symbol;
}

#define LOAD_HMMER(name) do {                                                  \
  void *p = lookup_symbol(#name, 1);                                           \
  _Static_assert(sizeof(real_##name) == sizeof(p), "pointer size mismatch");  \
  memcpy(&real_##name, &p, sizeof(p));                                         \
} while (0)

static void
resolve_symbols(void)
{
  Dl_info info;
  void *p;

  hmmer_handle = dlopen("liblibhmmer.so", RTLD_NOW | RTLD_NOLOAD);
  easel_handle = dlopen("liblibeasel.so", RTLD_NOW | RTLD_NOLOAD);
  LOAD_HMMER(p7_BackwardParser);
  LOAD_HMMER(p7_domaindef_ByPosteriorHeuristics);
  LOAD_HMMER(p7_DomainDecoding);
  LOAD_HMMER(p7_Forward);
  LOAD_HMMER(p7_Backward);
  LOAD_HMMER(p7_StochasticTrace);
  LOAD_HMMER(p7_spensemble_Cluster);
  LOAD_HMMER(p7_Decoding);
  LOAD_HMMER(p7_Null2_ByExpectation);
  LOAD_HMMER(p7_Null2_ByTrace);
  LOAD_HMMER(p7_OptimalAccuracy);
  LOAD_HMMER(p7_OATrace);
  LOAD_HMMER(p7_alidisplay_Create);
  LOAD_HMMER(p7_tophits_SortBySortkey);
  LOAD_HMMER(p7_tophits_Threshold);

  p = easel_handle == NULL ? NULL : dlsym(easel_handle,
                                          "esl_randomness_GetSeed");
  if (p == NULL) p = lookup_symbol("esl_randomness_GetSeed", 0);
  memcpy(&real_esl_randomness_GetSeed, &p, sizeof(p));

  if (dladdr((void *) real_p7_BackwardParser, &info) != 0 &&
      info.dli_fname != NULL) {
    size_t n = strnlen(info.dli_fname, sizeof(target_library) - 1);
    memcpy(target_library, info.dli_fname, n);
    target_library[n] = '\0';
  }
}

static uint64_t
elapsed_ns(const struct timespec *start, const struct timespec *end)
{
  int64_t seconds = (int64_t) end->tv_sec - (int64_t) start->tv_sec;
  int64_t nanoseconds = (int64_t) end->tv_nsec - (int64_t) start->tv_nsec;
  int64_t elapsed = seconds * INT64_C(1000000000) + nanoseconds;
  if (elapsed < 0) {
    atomic_fetch_add_explicit(&clock_errors, 1, memory_order_relaxed);
    return 0;
  }
  return (uint64_t) elapsed;
}

static uint64_t
finish_timer(enum metric_id id, int status, const struct timespec *start,
             int started)
{
  struct timespec end;
  uint64_t elapsed = 0;
  if (started && clock_gettime(CLOCK_MONOTONIC, &end) == 0)
    elapsed = elapsed_ns(start, &end);
  else
    atomic_fetch_add_explicit(&clock_errors, 1, memory_order_relaxed);

  atomic_fetch_add_explicit(&metrics[id].calls, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&metrics[id].elapsed_ns, elapsed,
                            memory_order_relaxed);
  atomic_fetch_add_explicit(status == eslOK ? &metrics[id].ok
                                            : &metrics[id].errors,
                            1, memory_order_relaxed);
  return elapsed;
}

static void
copy_name(char *dst, size_t size, const char *src)
{
  size_t i = 0;
  if (src != NULL) {
    while (i + 1 < size && src[i] != '\0') {
      unsigned char c = (unsigned char) src[i];
      dst[i] = (c == '\t' || c == '\n' || c == '\r') ? '_' : (char) c;
      i++;
    }
  }
  dst[i] = '\0';
}

static uint64_t
omx_bytes(const P7_OMX *ox)
{
  if (ox == NULL) return 0;
  return (uint64_t) sizeof(*ox) + (uint64_t) ox->ncells * 12u + 15u +
         (uint64_t) ox->allocR * 3u * sizeof(void *) +
         (uint64_t) ox->allocXR * p7X_NXCELLS * sizeof(float) + 15u;
}

static uint64_t
trace_bytes(const P7_TRACE *tr)
{
  uint64_t bytes;
  if (tr == NULL) return 0;
  bytes = sizeof(*tr) + (uint64_t) tr->nalloc *
          (sizeof(char) + 2u * sizeof(int) + (tr->pp ? sizeof(float) : 0u));
  bytes += (uint64_t) tr->ndomalloc * 6u * sizeof(int);
  return bytes;
}

static uint64_t
spensemble_bytes(const P7_SPENSEMBLE *sp)
{
  if (sp == NULL) return 0;
  return sizeof(*sp) +
         (uint64_t) sp->nalloc * sizeof(struct p7_spcoord_s) +
         (uint64_t) sp->nalloc * 3u * sizeof(int) +
         (uint64_t) sp->epc_alloc * sizeof(int) +
         (uint64_t) sp->nsigc_alloc * sizeof(struct p7_spcoord_s);
}

static uint64_t
domaindef_bytes(const P7_DOMAINDEF *ddef)
{
  if (ddef == NULL) return 0;
  return sizeof(*ddef) + (uint64_t) (ddef->Lalloc + 1) * 4u * sizeof(float) +
         (uint64_t) ddef->nalloc * sizeof(P7_DOMAIN) +
         spensemble_bytes(ddef->sp) + trace_bytes(ddef->tr) +
         trace_bytes(ddef->gtr);
}

static uint64_t
active_domain_bytes(const P7_DOMAINDEF *ddef)
{
  uint64_t bytes = 0;
  int d;
  if (ddef == NULL || ddef->dcl == NULL) return 0;
  bytes = (uint64_t) ddef->ndom * sizeof(P7_DOMAIN);
  for (d = 0; d < ddef->ndom; d++)
    if (ddef->dcl[d].ad != NULL)
      bytes += sizeof(P7_ALIDISPLAY) + (uint64_t) ddef->dcl[d].ad->memsize;
  return bytes;
}

static void
replay_regions(struct row *row, const P7_DOMAINDEF *ddef)
{
  int i = -1;
  int j;
  int triggered = 0;

  row->min_rt1_margin = FLT_MAX;
  row->min_start_rt2_margin = FLT_MAX;
  row->min_end_rt2_margin = FLT_MAX;
  row->min_rt3_margin = FLT_MAX;
  row->min_b_delta = FLT_MAX;
  row->min_e_delta = FLT_MAX;
  row->min_mocc = FLT_MAX;
  row->max_mocc = -FLT_MAX;

  for (j = 1; j <= row->L; j++) {
    float bdelta = ddef->btot[j] - ddef->btot[j-1];
    float edelta = ddef->etot[j] - ddef->etot[j-1];
    float mocc = ddef->mocc[j];
    if (!isfinite(bdelta) || !isfinite(edelta) || !isfinite(mocc)) {
      row->nonfinite_posterior++;
      continue;
    }
    if (bdelta < row->min_b_delta) row->min_b_delta = bdelta;
    if (edelta < row->min_e_delta) row->min_e_delta = edelta;
    if (mocc < row->min_mocc) row->min_mocc = mocc;
    if (mocc > row->max_mocc) row->max_mocc = mocc;

    if (!triggered) {
      float extent = mocc - bdelta;
      float margin = fabsf(extent - ddef->rt2);
      if (margin < row->min_start_rt2_margin)
        row->min_start_rt2_margin = margin;
      margin = fabsf(mocc - ddef->rt1);
      if (margin < row->min_rt1_margin) row->min_rt1_margin = margin;
      if (extent < ddef->rt2) i = j;
      else if (i == -1) i = j;
      if (mocc >= ddef->rt1) triggered = 1;
    } else {
      float extent = mocc - edelta;
      float margin = fabsf(extent - ddef->rt2);
      if (margin < row->min_end_rt2_margin)
        row->min_end_rt2_margin = margin;
      if (extent < ddef->rt2) {
        float max_expected = -1.0f;
        int z;
        int length = j - i + 1;
        row->replay_regions++;
        row->region_residues += (uint64_t) length;
        if (length > row->region_max) row->region_max = length;
        for (z = i; z <= j; z++) {
          float left = ddef->etot[z] - ddef->etot[i-1];
          float right = ddef->btot[j] - ddef->btot[z-1];
          float expected = left < right ? left : right;
          if (expected > max_expected) max_expected = expected;
        }
        margin = fabsf(max_expected - ddef->rt3);
        if (margin < row->min_rt3_margin) row->min_rt3_margin = margin;
        if (max_expected >= ddef->rt3) {
          row->replay_clustered++;
          row->clustered_region_residues += (uint64_t) length;
          if (length > row->clustered_region_max)
            row->clustered_region_max = length;
        }
        i = -1;
        triggered = 0;
      }
    }
  }
  row->unclosed_region = triggered;
  row->terminal_be_delta = fabsf(ddef->btot[row->L] - ddef->etot[row->L]);
  if (row->min_rt1_margin == FLT_MAX) row->min_rt1_margin = NAN;
  if (row->min_start_rt2_margin == FLT_MAX) row->min_start_rt2_margin = NAN;
  if (row->min_end_rt2_margin == FLT_MAX) row->min_end_rt2_margin = NAN;
  if (row->min_rt3_margin == FLT_MAX) row->min_rt3_margin = NAN;
}

static void
append_row(struct row *row)
{
  pthread_mutex_lock(&rows_lock);
  if (row_count == row_capacity) {
    size_t capacity = row_capacity == 0 ? 512 : row_capacity * 2;
    void *p = realloc(rows, capacity * sizeof(*rows));
    if (p == NULL) {
      atomic_fetch_add_explicit(&dropped_rows, 1, memory_order_relaxed);
      pthread_mutex_unlock(&rows_lock);
      return;
    }
    rows = p;
    row_capacity = capacity;
  }
  rows[row_count++] = *row;
  pthread_mutex_unlock(&rows_lock);
}

__attribute__((constructor)) static void
initialize(void)
{
  const char *path = getenv("PLAN7_POST_FORWARD_CENSUS");
  size_t n;
  origin_pid = getpid();
  if (path == NULL || path[0] == '\0') return;
  n = strnlen(path, sizeof(output_path));
  if (n == sizeof(output_path)) return;
  memcpy(output_path, path, n + 1);
  enabled = 1;
}

int
p7_BackwardParser(const ESL_DSQ *dsq, int L, const P7_OPROFILE *om,
                  const P7_OMX *fwd, P7_OMX *bck, float *ret_sc)
{
  struct timespec start;
  int started;
  int status;
  uint64_t elapsed;
  pthread_once(&resolve_once, resolve_symbols);
  started = enabled && clock_gettime(CLOCK_MONOTONIC, &start) == 0;
  status = real_p7_BackwardParser(dsq, L, om, fwd, bck, ret_sc);
  if (!enabled) return status;
  elapsed = finish_timer(METRIC_BACKWARD_PARSER, status, &start, started);
  tls.backward_dsq = dsq;
  tls.backward_om = om;
  tls.backward_L = L;
  tls.backward_status = status;
  tls.backward_ns = elapsed;
  return status;
}

int
p7_DomainDecoding(const P7_OPROFILE *om, const P7_OMX *fwd,
                  const P7_OMX *bck, P7_DOMAINDEF *ddef)
{
  struct timespec start;
  int started;
  int status;
  uint64_t elapsed;
  pthread_once(&resolve_once, resolve_symbols);
  started = enabled && clock_gettime(CLOCK_MONOTONIC, &start) == 0;
  status = real_p7_DomainDecoding(om, fwd, bck, ddef);
  if (!enabled) return status;
  elapsed = finish_timer(METRIC_DOMAIN_DECODING, status, &start, started);
  if (tls.in_domain) {
    tls.decoding_calls++;
    tls.decoding_ns += elapsed;
  }
  return status;
}

#define DEFINE_TIMED_INT(function, metric, declaration, invocation)            \
int function declaration                                                       \
{                                                                              \
  struct timespec probe_start;                                                 \
  int started;                                                                 \
  int status;                                                                  \
  pthread_once(&resolve_once, resolve_symbols);                                \
  started = enabled && clock_gettime(CLOCK_MONOTONIC, &probe_start) == 0;     \
  status = real_##function invocation;                                         \
  if (enabled)                                                                 \
    (void) finish_timer((metric), status, &probe_start, started);              \
  return status;                                                               \
}

DEFINE_TIMED_INT(p7_Forward, METRIC_FORWARD,
  (const ESL_DSQ *dsq, int L, const P7_OPROFILE *om, P7_OMX *mx, float *sc),
  (dsq, L, om, mx, sc))

DEFINE_TIMED_INT(p7_Backward, METRIC_BACKWARD,
  (const ESL_DSQ *dsq, int L, const P7_OPROFILE *om, const P7_OMX *fwd,
   P7_OMX *bck, float *sc), (dsq, L, om, fwd, bck, sc))

int
p7_StochasticTrace(ESL_RANDOMNESS *rng, const ESL_DSQ *dsq, int L,
                   const P7_OPROFILE *om, const P7_OMX *mx, P7_TRACE *tr)
{
  struct timespec start;
  int started;
  int status;
  pthread_once(&resolve_once, resolve_symbols);
  started = enabled && clock_gettime(CLOCK_MONOTONIC, &start) == 0;
  status = real_p7_StochasticTrace(rng, dsq, L, om, mx, tr);
  if (enabled) {
    (void) finish_timer(METRIC_STOCHASTIC_TRACE, status, &start, started);
    if (tls.in_domain) {
      tls.stochastic_calls++;
      if (status != eslOK) tls.stochastic_failures++;
    }
  }
  return status;
}

int
p7_spensemble_Cluster(P7_SPENSEMBLE *sp, float min_overlap, int of_smaller,
                      int max_diagdiff, float min_posterior,
                      float min_endpointp, int *ret_nclusters)
{
  struct timespec start;
  int started;
  int status;
  pthread_once(&resolve_once, resolve_symbols);
  started = enabled && clock_gettime(CLOCK_MONOTONIC, &start) == 0;
  if (enabled && tls.in_domain) tls.sampled_segments += (uint64_t) sp->n;
  status = real_p7_spensemble_Cluster(sp, min_overlap, of_smaller, max_diagdiff,
                                      min_posterior, min_endpointp,
                                      ret_nclusters);
  if (enabled) {
    (void) finish_timer(METRIC_CLUSTER, status, &start, started);
    if (tls.in_domain && status == eslOK && ret_nclusters != NULL)
      tls.raw_clusters += (uint64_t) *ret_nclusters;
  }
  return status;
}

DEFINE_TIMED_INT(p7_Decoding, METRIC_DECODING,
  (const P7_OPROFILE *om, const P7_OMX *fwd, P7_OMX *bck, P7_OMX *pp),
  (om, fwd, bck, pp))

DEFINE_TIMED_INT(p7_Null2_ByExpectation, METRIC_NULL2_EXPECTATION,
  (const P7_OPROFILE *om, const P7_OMX *pp, float *null2),
  (om, pp, null2))

DEFINE_TIMED_INT(p7_Null2_ByTrace, METRIC_NULL2_TRACE,
  (const P7_OPROFILE *om, const P7_TRACE *tr, int start, int end,
   P7_OMX *wrk, float *null2), (om, tr, start, end, wrk, null2))

DEFINE_TIMED_INT(p7_OptimalAccuracy, METRIC_OPTIMAL_ACCURACY,
  (const P7_OPROFILE *om, const P7_OMX *pp, P7_OMX *mx, float *ret_e),
  (om, pp, mx, ret_e))

DEFINE_TIMED_INT(p7_OATrace, METRIC_OA_TRACE,
  (const P7_OPROFILE *om, const P7_OMX *pp, const P7_OMX *mx, P7_TRACE *tr),
  (om, pp, mx, tr))

P7_ALIDISPLAY *
p7_alidisplay_Create(const P7_TRACE *tr, int which, const P7_OPROFILE *om,
                     const ESL_SQ *sq, const ESL_SQ *ntsq)
{
  struct timespec start;
  int started;
  P7_ALIDISPLAY *ad;
  pthread_once(&resolve_once, resolve_symbols);
  started = enabled && clock_gettime(CLOCK_MONOTONIC, &start) == 0;
  ad = real_p7_alidisplay_Create(tr, which, om, sq, ntsq);
  if (enabled)
    (void) finish_timer(METRIC_ALIDISPLAY, ad == NULL ? eslFAIL : eslOK,
                        &start, started);
  return ad;
}

DEFINE_TIMED_INT(p7_tophits_SortBySortkey, METRIC_TOPHITS_SORT,
  (P7_TOPHITS *hits), (hits))

DEFINE_TIMED_INT(p7_tophits_Threshold, METRIC_TOPHITS_THRESHOLD,
  (P7_TOPHITS *hits, P7_PIPELINE *pipeline), (hits, pipeline))

int
p7_domaindef_ByPosteriorHeuristics(
    const ESL_SQ *sq, const ESL_SQ *ntsq, P7_OPROFILE *om, P7_OMX *oxf,
    P7_OMX *oxb, P7_OMX *fwd, P7_OMX *bck, P7_DOMAINDEF *ddef, P7_BG *bg,
    int long_target, P7_BG *bg_tmp, float *scores, float *forward_emissions)
{
  struct timespec start;
  struct row row;
  int started;
  int status;
  int d;

  pthread_once(&resolve_once, resolve_symbols);
  if (!enabled)
    return real_p7_domaindef_ByPosteriorHeuristics(
        sq, ntsq, om, oxf, oxb, fwd, bck, ddef, bg, long_target, bg_tmp,
        scores, forward_emissions);

  memset(&row, 0, sizeof(row));
  tls.in_domain = 1;
  tls.decoding_calls = 0;
  tls.decoding_ns = 0;
  tls.stochastic_calls = 0;
  tls.stochastic_failures = 0;
  tls.sampled_segments = 0;
  tls.raw_clusters = 0;
  started = clock_gettime(CLOCK_MONOTONIC, &start) == 0;
  status = real_p7_domaindef_ByPosteriorHeuristics(
      sq, ntsq, om, oxf, oxb, fwd, bck, ddef, bg, long_target, bg_tmp, scores,
      forward_emissions);
  tls.in_domain = 0;

  row.duration_ns = finish_timer(METRIC_DOMAIN_WORKFLOW, status, &start,
                                 started);
  row.serial = atomic_fetch_add_explicit(&next_serial, 1, memory_order_relaxed);
  row.sequence_idx = sq->idx;
  copy_name(row.sequence_name, sizeof(row.sequence_name), sq->name);
  copy_name(row.profile_name, sizeof(row.profile_name), om->name);
  row.L = (int) sq->n;
  row.M = om->M;
  row.status = status;
  if (tls.backward_dsq == sq->dsq && tls.backward_om == om &&
      tls.backward_L == sq->n) {
    row.backward_status = tls.backward_status;
    row.backward_ns = tls.backward_ns;
  } else {
    row.backward_status = INT32_MIN;
  }
  row.decoding_calls = tls.decoding_calls;
  row.decoding_ns = tls.decoding_ns;
  row.rng_seed = real_esl_randomness_GetSeed(ddef->r);
  row.do_reseeding = ddef->do_reseeding;
  row.nsamples = ddef->nsamples;
  row.nexpected = ddef->nexpected;
  row.rt1 = ddef->rt1;
  row.rt2 = ddef->rt2;
  row.rt3 = ddef->rt3;
  row.nregions = ddef->nregions;
  row.nclustered = ddef->nclustered;
  row.noverlaps = ddef->noverlaps;
  row.nenvelopes = ddef->nenvelopes;
  row.ndom = ddef->ndom;
  row.stochastic_calls = tls.stochastic_calls;
  row.stochastic_failures = tls.stochastic_failures;
  row.sampled_segments = tls.sampled_segments;
  row.raw_clusters = tls.raw_clusters;
  row.simple_regions = ddef->nregions - ddef->nclustered;
  row.final_cluster_envelopes = ddef->nenvelopes - row.simple_regions;
  if (status == eslOK) replay_regions(&row, ddef);
  for (d = 0; d < ddef->ndom; d++) {
    int env_length = (int) (ddef->dcl[d].jenv - ddef->dcl[d].ienv + 1);
    int ali_length = (int) (ddef->dcl[d].jali - ddef->dcl[d].iali + 1);
    row.envelope_residues += (uint64_t) env_length;
    row.alignment_residues += (uint64_t) ali_length;
    if (env_length > row.envelope_max) row.envelope_max = env_length;
    if (ali_length > row.alignment_max) row.alignment_max = ali_length;
  }
  row.oxf_bytes = omx_bytes(oxf);
  row.oxb_bytes = omx_bytes(oxb);
  row.fwd_bytes = omx_bytes(fwd);
  row.bck_bytes = omx_bytes(bck);
  row.ddef_bytes = domaindef_bytes(ddef);
  row.active_domain_bytes = active_domain_bytes(ddef);
  append_row(&row);
  return status;
}

static void
write_float(FILE *out, float value)
{
  if (isnan(value)) fputs("nan", out);
  else fprintf(out, "%.9g", value);
}

__attribute__((destructor)) static void
report(void)
{
  FILE *out;
  size_t i;
  if (!enabled || getpid() != origin_pid || row_count == 0) return;
  out = fopen(output_path, "w");
  if (out == NULL) return;
  fprintf(out, "#schema\tplan7_post_forward_census\t1\n");
  fprintf(out, "#target_library\t%s\n", target_library);
  fprintf(out, "#rows\t%zu\n", row_count);
  fprintf(out, "#dropped_rows\t%" PRIuFAST64 "\n",
          atomic_load_explicit(&dropped_rows, memory_order_relaxed));
  fprintf(out, "#clock_errors\t%" PRIuFAST64 "\n",
          atomic_load_explicit(&clock_errors, memory_order_relaxed));
  for (i = 0; i < METRIC_COUNT; i++)
    fprintf(out, "#metric\t%s\t%" PRIuFAST64 "\t%" PRIuFAST64
                 "\t%" PRIuFAST64 "\t%" PRIuFAST64 "\n",
            metrics[i].name,
            atomic_load_explicit(&metrics[i].calls, memory_order_relaxed),
            atomic_load_explicit(&metrics[i].elapsed_ns, memory_order_relaxed),
            atomic_load_explicit(&metrics[i].ok, memory_order_relaxed),
            atomic_load_explicit(&metrics[i].errors, memory_order_relaxed));
  fputs("serial\tsequence_idx\tsequence_name\tprofile_name\tL\tM\tstatus\t"
        "duration_ns\tbackward_status\tbackward_ns\tdecoding_calls\t"
        "decoding_ns\trng_seed\tdo_reseeding\tnsamples\tnexpected\trt1\t"
        "rt2\trt3\tnregions\treplay_regions\tnclustered\treplay_clustered\t"
        "noverlaps\tnenvelopes\tndom\tstochastic_calls\t"
        "stochastic_failures\tsampled_segments\traw_clusters\tsimple_regions\t"
        "final_cluster_envelopes\tregion_residues\tregion_max\t"
        "clustered_region_residues\tclustered_region_max\tenvelope_residues\t"
        "envelope_max\talignment_residues\talignment_max\tmin_rt1_margin\t"
        "min_start_rt2_margin\tmin_end_rt2_margin\tmin_rt3_margin\t"
        "min_b_delta\tmin_e_delta\tmin_mocc\tmax_mocc\tterminal_be_delta\t"
        "nonfinite_posterior\tunclosed_region\toxf_bytes\toxb_bytes\t"
        "fwd_bytes\tbck_bytes\tddef_bytes\tactive_domain_bytes\n", out);
  for (i = 0; i < row_count; i++) {
    const struct row *r = &rows[i];
    fprintf(out, "%" PRIu64 "\t%" PRId64 "\t%s\t%s\t%d\t%d\t%d\t%"
                 PRIu64 "\t%d\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64
                 "\t%" PRIu32 "\t%d\t%d\t",
            r->serial, r->sequence_idx, r->sequence_name, r->profile_name,
            r->L, r->M, r->status, r->duration_ns, r->backward_status,
            r->backward_ns, r->decoding_calls, r->decoding_ns, r->rng_seed,
            r->do_reseeding, r->nsamples);
    write_float(out, r->nexpected);
    fputc('\t', out); write_float(out, r->rt1);
    fputc('\t', out); write_float(out, r->rt2);
    fputc('\t', out); write_float(out, r->rt3);
    fprintf(out, "\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%" PRIu64
                 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%d\t%d\t%"
                 PRIu64 "\t%d\t%" PRIu64 "\t%d\t%" PRIu64 "\t%d\t%"
                 PRIu64 "\t%d\t",
            r->nregions, r->replay_regions, r->nclustered,
            r->replay_clustered, r->noverlaps, r->nenvelopes, r->ndom,
            r->stochastic_calls, r->stochastic_failures, r->sampled_segments,
            r->raw_clusters, r->simple_regions, r->final_cluster_envelopes,
            r->region_residues, r->region_max, r->clustered_region_residues,
            r->clustered_region_max, r->envelope_residues, r->envelope_max,
            r->alignment_residues, r->alignment_max);
    write_float(out, r->min_rt1_margin);
    fputc('\t', out); write_float(out, r->min_start_rt2_margin);
    fputc('\t', out); write_float(out, r->min_end_rt2_margin);
    fputc('\t', out); write_float(out, r->min_rt3_margin);
    fputc('\t', out); write_float(out, r->min_b_delta);
    fputc('\t', out); write_float(out, r->min_e_delta);
    fputc('\t', out); write_float(out, r->min_mocc);
    fputc('\t', out); write_float(out, r->max_mocc);
    fputc('\t', out); write_float(out, r->terminal_be_delta);
    fprintf(out, "\t%" PRIu64 "\t%d\t%" PRIu64 "\t%" PRIu64 "\t%"
                 PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\n",
            r->nonfinite_posterior, r->unclosed_region, r->oxf_bytes,
            r->oxb_bytes, r->fwd_bytes, r->bck_bytes, r->ddef_bytes,
            r->active_domain_bytes);
  }
  (void) fclose(out);
  free(rows);
}
