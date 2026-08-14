#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/*
 * Observer-only LD_PRELOAD shim for Astra's bundled HMMER library.
 *
 * The signatures below deliberately use opaque pointers while preserving the
 * exact HMMER 3.4 ABI.  This keeps the probe independent of Astra/PyHMMER's
 * private include paths.  No data behind any pointer is read or modified.
 */

enum metric_id {
  METRIC_PIPELINE = 0,
  METRIC_NULL1,
  METRIC_MSV,
  METRIC_SSV,
  METRIC_BIAS,
  METRIC_VITERBI,
  METRIC_FORWARD_PARSER,
  METRIC_BACKWARD_PARSER,
  METRIC_DOMAIN_WORKFLOW,
  METRIC_FORWARD,
  METRIC_BACKWARD,
  METRIC_DOMAIN_DECODING,
  METRIC_COUNT
};

struct metric {
  const char *name;
  atomic_uint_fast64_t calls;
  atomic_uint_fast64_t elapsed_ns;
  atomic_uint_fast64_t status_ok;
  atomic_uint_fast64_t status_erange;
  atomic_uint_fast64_t status_noresult;
  atomic_uint_fast64_t status_other;
};

#define METRIC_INIT(label)                                                     \
  { (label), ATOMIC_VAR_INIT(0), ATOMIC_VAR_INIT(0), ATOMIC_VAR_INIT(0),      \
    ATOMIC_VAR_INIT(0), ATOMIC_VAR_INIT(0), ATOMIC_VAR_INIT(0) }

static struct metric metrics[METRIC_COUNT] = {
    METRIC_INIT("p7_Pipeline"),
    METRIC_INIT("p7_bg_NullOne"),
    METRIC_INIT("p7_MSVFilter"),
    METRIC_INIT("p7_SSVFilter"),
    METRIC_INIT("p7_bg_FilterScore"),
    METRIC_INIT("p7_ViterbiFilter"),
    METRIC_INIT("p7_ForwardParser"),
    METRIC_INIT("p7_BackwardParser"),
    METRIC_INIT("p7_domaindef_ByPosteriorHeuristics"),
    METRIC_INIT("p7_Forward"),
    METRIC_INIT("p7_Backward"),
    METRIC_INIT("p7_DomainDecoding"),
};

/* Stable Easel status values used by the bundled HMMER 3.4 ABI. */
enum {
  PROBE_ESL_OK = 0,
  PROBE_ESL_ERANGE = 16,
  PROBE_ESL_ENORESULT = 19,
};

static int probe_enabled;
static char probe_path[PATH_MAX];
static char target_library[PATH_MAX] = "unknown";
static pid_t origin_pid;
static atomic_uint_fast64_t clock_errors = ATOMIC_VAR_INIT(0);
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static void *hmmer_handle;

typedef int (*pipeline_fn)(void *, void *, void *, const void *, const void *,
                           void *);
typedef int (*filter_fn)(const unsigned char *, int, const void *, void *,
                         float *);
typedef int (*ssv_fn)(const unsigned char *, int, const void *, float *);
typedef int (*null1_fn)(const void *, const unsigned char *, int, float *);
typedef int (*bias_fn)(void *, const unsigned char *, int, float *);
typedef int (*backward_fn)(const unsigned char *, int, const void *,
                           const void *, void *, float *);
typedef int (*domain_decoding_fn)(const void *, const void *, const void *,
                                  void *);
typedef int (*domain_workflow_fn)(const void *, const void *, void *, void *,
                                  void *, void *, void *, void *, void *, int,
                                  void *, float *, float *);

static pipeline_fn real_p7_Pipeline;
static null1_fn real_p7_bg_NullOne;
static filter_fn real_p7_MSVFilter;
static ssv_fn real_p7_SSVFilter;
static bias_fn real_p7_bg_FilterScore;
static filter_fn real_p7_ViterbiFilter;
static filter_fn real_p7_ForwardParser;
static backward_fn real_p7_BackwardParser;
static domain_workflow_fn real_p7_domaindef_ByPosteriorHeuristics;
static filter_fn real_p7_Forward;
static backward_fn real_p7_Backward;
static domain_decoding_fn real_p7_DomainDecoding;

static void
resolution_failure(const char *symbol)
{
  char message[256];
  int n = snprintf(message, sizeof(message),
                   "astra_stage_probe: cannot resolve %s\n", symbol);
  if (n > 0) {
    size_t length = (size_t)n < sizeof(message) ? (size_t)n : sizeof(message);
    ssize_t write_status = write(STDERR_FILENO, message, length);
    (void)write_status;
  }
  _exit(127);
}

static void *
lookup_hmmer_symbol(const char *name)
{
  void *symbol = NULL;

  if (hmmer_handle != NULL) symbol = dlsym(hmmer_handle, name);
  if (symbol == NULL)       symbol = dlsym(RTLD_NEXT, name);
  if (symbol == NULL)       resolution_failure(name);
  return symbol;
}

#define LOAD_SYMBOL(name)                                                      \
  do {                                                                         \
    void *symbol_address = lookup_hmmer_symbol(#name);                          \
    _Static_assert(sizeof(real_##name) == sizeof(symbol_address),               \
                   "function and data pointers differ on this platform");      \
    memcpy(&real_##name, &symbol_address, sizeof(symbol_address));              \
  } while (0)

static void
resolve_symbols(void)
{
  Dl_info info;

  hmmer_handle = dlopen("liblibhmmer.so", RTLD_NOW | RTLD_NOLOAD);

  LOAD_SYMBOL(p7_Pipeline);
  LOAD_SYMBOL(p7_bg_NullOne);
  LOAD_SYMBOL(p7_MSVFilter);
  LOAD_SYMBOL(p7_SSVFilter);
  LOAD_SYMBOL(p7_bg_FilterScore);
  LOAD_SYMBOL(p7_ViterbiFilter);
  LOAD_SYMBOL(p7_ForwardParser);
  LOAD_SYMBOL(p7_BackwardParser);
  LOAD_SYMBOL(p7_domaindef_ByPosteriorHeuristics);
  LOAD_SYMBOL(p7_Forward);
  LOAD_SYMBOL(p7_Backward);
  LOAD_SYMBOL(p7_DomainDecoding);

  if (dladdr((void *)real_p7_Pipeline, &info) != 0 && info.dli_fname != NULL) {
    size_t length = strnlen(info.dli_fname, sizeof(target_library) - 1);
    memcpy(target_library, info.dli_fname, length);
    target_library[length] = '\0';
  }
}

__attribute__((constructor)) static void
probe_initialize(void)
{
  const char *path = getenv("PLAN7_ASTRA_STAGE_PROBE");

  origin_pid = getpid();
  if (path == NULL || path[0] == '\0') return;

  size_t length = strnlen(path, sizeof(probe_path));
  if (length == sizeof(probe_path)) return;

  memcpy(probe_path, path, length + 1);
  probe_enabled = 1;
}

static void
record_metric(enum metric_id id, int status, const struct timespec *start,
              const struct timespec *end, int clock_ok)
{
  struct metric *metric = &metrics[id];
  atomic_fetch_add_explicit(&metric->calls, 1, memory_order_relaxed);

  switch (status) {
  case PROBE_ESL_OK:
    atomic_fetch_add_explicit(&metric->status_ok, 1, memory_order_relaxed);
    break;
  case PROBE_ESL_ERANGE:
    atomic_fetch_add_explicit(&metric->status_erange, 1,
                              memory_order_relaxed);
    break;
  case PROBE_ESL_ENORESULT:
    atomic_fetch_add_explicit(&metric->status_noresult, 1,
                              memory_order_relaxed);
    break;
  default:
    atomic_fetch_add_explicit(&metric->status_other, 1,
                              memory_order_relaxed);
    break;
  }

  if (clock_ok) {
    int64_t seconds = (int64_t)end->tv_sec - (int64_t)start->tv_sec;
    int64_t nanoseconds = (int64_t)end->tv_nsec - (int64_t)start->tv_nsec;
    int64_t elapsed = seconds * INT64_C(1000000000) + nanoseconds;
    if (elapsed >= 0) {
      atomic_fetch_add_explicit(&metric->elapsed_ns, (uint64_t)elapsed,
                                memory_order_relaxed);
      return;
    }
  }
  atomic_fetch_add_explicit(&clock_errors, 1, memory_order_relaxed);
}

#define OBSERVED_CALL(metric_id, expression)                                   \
  do {                                                                         \
    struct timespec probe_start;                                                \
    struct timespec probe_end;                                                  \
    int start_ok = clock_gettime(CLOCK_MONOTONIC, &probe_start) == 0;           \
    int probe_status = (expression);                                            \
    int end_ok = clock_gettime(CLOCK_MONOTONIC, &probe_end) == 0;               \
    record_metric((metric_id), probe_status, &probe_start, &probe_end,           \
                  start_ok && end_ok);                                          \
    return probe_status;                                                        \
  } while (0)

int
p7_Pipeline(void *pipeline, void *profile, void *background,
            const void *sequence, const void *translated_sequence, void *hits)
{
  pthread_once(&resolve_once, resolve_symbols);
  if (!probe_enabled)
    return real_p7_Pipeline(pipeline, profile, background, sequence,
                            translated_sequence, hits);
  OBSERVED_CALL(METRIC_PIPELINE,
                real_p7_Pipeline(pipeline, profile, background, sequence,
                                 translated_sequence, hits));
}

int
p7_bg_NullOne(const void *background, const unsigned char *sequence, int length,
              float *score)
{
  pthread_once(&resolve_once, resolve_symbols);
  if (!probe_enabled)
    return real_p7_bg_NullOne(background, sequence, length, score);
  OBSERVED_CALL(METRIC_NULL1,
                real_p7_bg_NullOne(background, sequence, length, score));
}

int
p7_MSVFilter(const unsigned char *sequence, int length, const void *profile,
             void *matrix, float *score)
{
  pthread_once(&resolve_once, resolve_symbols);
  if (!probe_enabled)
    return real_p7_MSVFilter(sequence, length, profile, matrix, score);
  OBSERVED_CALL(METRIC_MSV,
                real_p7_MSVFilter(sequence, length, profile, matrix, score));
}

int
p7_SSVFilter(const unsigned char *sequence, int length, const void *profile,
             float *score)
{
  pthread_once(&resolve_once, resolve_symbols);
  if (!probe_enabled)
    return real_p7_SSVFilter(sequence, length, profile, score);
  OBSERVED_CALL(METRIC_SSV,
                real_p7_SSVFilter(sequence, length, profile, score));
}

int
p7_bg_FilterScore(void *background, const unsigned char *sequence, int length,
                  float *score)
{
  pthread_once(&resolve_once, resolve_symbols);
  if (!probe_enabled)
    return real_p7_bg_FilterScore(background, sequence, length, score);
  OBSERVED_CALL(METRIC_BIAS,
                real_p7_bg_FilterScore(background, sequence, length, score));
}

int
p7_ViterbiFilter(const unsigned char *sequence, int length, const void *profile,
                 void *matrix, float *score)
{
  pthread_once(&resolve_once, resolve_symbols);
  if (!probe_enabled)
    return real_p7_ViterbiFilter(sequence, length, profile, matrix, score);
  OBSERVED_CALL(METRIC_VITERBI,
                real_p7_ViterbiFilter(sequence, length, profile, matrix,
                                      score));
}

int
p7_ForwardParser(const unsigned char *sequence, int length, const void *profile,
                 void *matrix, float *score)
{
  pthread_once(&resolve_once, resolve_symbols);
  if (!probe_enabled)
    return real_p7_ForwardParser(sequence, length, profile, matrix, score);
  OBSERVED_CALL(METRIC_FORWARD_PARSER,
                real_p7_ForwardParser(sequence, length, profile, matrix,
                                      score));
}

int
p7_BackwardParser(const unsigned char *sequence, int length,
                  const void *profile, const void *forward_matrix,
                  void *backward_matrix, float *score)
{
  pthread_once(&resolve_once, resolve_symbols);
  if (!probe_enabled)
    return real_p7_BackwardParser(sequence, length, profile, forward_matrix,
                                  backward_matrix, score);
  OBSERVED_CALL(METRIC_BACKWARD_PARSER,
                real_p7_BackwardParser(sequence, length, profile,
                                       forward_matrix, backward_matrix,
                                       score));
}

int
p7_domaindef_ByPosteriorHeuristics(
    const void *sequence, const void *translated_sequence, void *profile,
    void *forward_parser_matrix, void *backward_parser_matrix,
    void *forward_matrix, void *backward_matrix, void *domain_definition,
    void *background, int long_target, void *temporary_background,
    float *scores, float *forward_emissions)
{
  pthread_once(&resolve_once, resolve_symbols);
  if (!probe_enabled)
    return real_p7_domaindef_ByPosteriorHeuristics(
        sequence, translated_sequence, profile, forward_parser_matrix,
        backward_parser_matrix, forward_matrix, backward_matrix,
        domain_definition, background, long_target, temporary_background,
        scores, forward_emissions);
  OBSERVED_CALL(
      METRIC_DOMAIN_WORKFLOW,
      real_p7_domaindef_ByPosteriorHeuristics(
          sequence, translated_sequence, profile, forward_parser_matrix,
          backward_parser_matrix, forward_matrix, backward_matrix,
          domain_definition, background, long_target, temporary_background,
          scores, forward_emissions));
}

int
p7_Forward(const unsigned char *sequence, int length, const void *profile,
           void *matrix, float *score)
{
  pthread_once(&resolve_once, resolve_symbols);
  if (!probe_enabled)
    return real_p7_Forward(sequence, length, profile, matrix, score);
  OBSERVED_CALL(METRIC_FORWARD,
                real_p7_Forward(sequence, length, profile, matrix, score));
}

int
p7_Backward(const unsigned char *sequence, int length, const void *profile,
            const void *forward_matrix, void *backward_matrix, float *score)
{
  pthread_once(&resolve_once, resolve_symbols);
  if (!probe_enabled)
    return real_p7_Backward(sequence, length, profile, forward_matrix,
                            backward_matrix, score);
  OBSERVED_CALL(METRIC_BACKWARD,
                real_p7_Backward(sequence, length, profile, forward_matrix,
                                 backward_matrix, score));
}

int
p7_DomainDecoding(const void *profile, const void *forward_matrix,
                  const void *backward_matrix, void *domain_definition)
{
  pthread_once(&resolve_once, resolve_symbols);
  if (!probe_enabled)
    return real_p7_DomainDecoding(profile, forward_matrix, backward_matrix,
                                  domain_definition);
  OBSERVED_CALL(METRIC_DOMAIN_DECODING,
                real_p7_DomainDecoding(profile, forward_matrix,
                                       backward_matrix, domain_definition));
}

__attribute__((destructor)) static void
probe_report(void)
{
  FILE *output;

  if (!probe_enabled || getpid() != origin_pid) return;
  if (atomic_load_explicit(&metrics[METRIC_PIPELINE].calls,
                           memory_order_relaxed) == 0)
    return;

  output = fopen(probe_path, "w");
  if (output == NULL) return;

  fprintf(output, "#schema\tplan7_astra_stage_probe\t1\n");
  fprintf(output, "#clock\tCLOCK_MONOTONIC\n");
  fprintf(output, "#aggregation\tprocess_wide_inclusive\n");
  fprintf(output, "#target_library\t%s\n", target_library);
  fprintf(output,
          "#observer_overhead\ttwo_clock_gettime_calls_and_relaxed_atomic_"
          "updates_per_observed_call\n");
  fprintf(output, "#clock_errors\t%" PRIuFAST64 "\n",
          atomic_load_explicit(&clock_errors, memory_order_relaxed));
  fprintf(output,
          "stage\tcalls\telapsed_ns\tstatus_ok\tstatus_erange\t"
          "status_noresult\tstatus_other\n");

  for (int i = 0; i < METRIC_COUNT; ++i) {
    const struct metric *metric = &metrics[i];
    fprintf(output,
            "%s\t%" PRIuFAST64 "\t%" PRIuFAST64 "\t%" PRIuFAST64
            "\t%" PRIuFAST64 "\t%" PRIuFAST64 "\t%" PRIuFAST64 "\n",
            metric->name,
            atomic_load_explicit(&metric->calls, memory_order_relaxed),
            atomic_load_explicit(&metric->elapsed_ns, memory_order_relaxed),
            atomic_load_explicit(&metric->status_ok, memory_order_relaxed),
            atomic_load_explicit(&metric->status_erange,
                                 memory_order_relaxed),
            atomic_load_explicit(&metric->status_noresult,
                                 memory_order_relaxed),
            atomic_load_explicit(&metric->status_other,
                                 memory_order_relaxed));
  }

  (void)fclose(output);
}
