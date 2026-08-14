/* Independent byte-MSV oracle against pristine HMMER 3.4.
 *
 * The scalar recurrence intentionally does not reuse HMMER's striped SIMD
 * implementation. It consumes the canonical byte costs extracted from a
 * P7_OPROFILE, preserving unsigned saturating operation order exactly.
 */

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "easel.h"
#include "esl_gumbel.h"
#include "esl_sqio.h"
#include "hmmer.h"

#define ORACLE_SCHEMA_VERSION 1
#define HMMER_REVISION "hmmer-3.4:9acd8b6758a0ca5d21db6d167e0277484341929b"
#define HMMER_PIPELINE_MAX_TARGET_LENGTH 100000

typedef struct {
  int      status;
  uint8_t  xj;
  int      overflow_row;
  uint8_t  overflow_xe;
  float    score;
} SCALAR_MSV_RESULT;

typedef struct {
  P7_HMM **items;
  size_t   count;
  size_t   capacity;
} HMM_LIST;

typedef struct {
  ESL_SQ **items;
  size_t   count;
  size_t   capacity;
} SEQUENCE_LIST;

static void
fail(const char *message)
{
  fprintf(stderr, "msv-oracle: %s\n", message);
  exit(EXIT_FAILURE);
}

static void *
xrealloc(void *pointer, size_t bytes)
{
  void *resized = realloc(pointer, bytes);
  if (resized == NULL) fail("memory allocation failed");
  return resized;
}

static size_t
checked_allocation_size(size_t count, size_t element_size)
{
  if (element_size != 0 && count > SIZE_MAX / element_size)
    fail("allocation size overflow");
  return count * element_size;
}

static uint8_t
sat_add_u8(uint8_t left, uint8_t right)
{
  unsigned int sum = (unsigned int) left + (unsigned int) right;
  return (sum > UINT8_MAX) ? UINT8_MAX : (uint8_t) sum;
}

static uint8_t
sat_sub_u8(uint8_t left, uint8_t right)
{
  return (left < right) ? 0 : (uint8_t) (left - right);
}

static SCALAR_MSV_RESULT
scalar_full_msv(const ESL_DSQ *dsq, int length, const P7_OPROFILE *om,
                const uint8_t *emission_costs)
{
  SCALAR_MSV_RESULT result = { eslOK, 0, 0, 0, -eslINFINITY };
  uint8_t          *dp;
  uint8_t           xj = 0;
  uint8_t           transition_cost = (uint8_t) (om->tjb_b + om->tbm_b);
  uint8_t           xb = sat_sub_u8(om->base_b, transition_cost);
  int               i;
  int               k;

  dp = calloc((size_t) om->M + 1, sizeof(*dp));
  if (dp == NULL) fail("memory allocation failed in scalar MSV");

  for (i = 1; i <= length; i++) {
    uint8_t previous_diagonal = 0;
    uint8_t xe = 0;
    ESL_DSQ residue = dsq[i];

    if ((size_t) residue >= (size_t) om->abc->Kp)
      fail("digital residue code is outside the optimized-profile alphabet");

    for (k = 1; k <= om->M; k++) {
      uint8_t previous_row = dp[k];
      uint8_t score = (previous_diagonal > xb) ? previous_diagonal : xb;
      score = sat_add_u8(score, om->bias_b);
      score = sat_sub_u8(score, emission_costs[(size_t) om->abc->Kp * (size_t) k + residue]);
      dp[k] = score;
      if (score > xe) xe = score;
      previous_diagonal = previous_row;
    }

    if (sat_add_u8(xe, om->bias_b) == UINT8_MAX) {
      result.status = eslERANGE;
      result.overflow_row = i;
      result.overflow_xe = xe;
      result.score = eslINFINITY;
      free(dp);
      return result;
    }

    xe = sat_sub_u8(xe, om->tec_b);
    if (xe > xj) xj = xe;
    xb = sat_sub_u8((xj > om->base_b) ? xj : om->base_b, transition_cost);
  }

  result.xj = xj;
  result.score = ((float) ((int) xj - (int) om->tjb_b) - (float) om->base_b);
  result.score /= om->scale_b;
  result.score -= 3.0f;
  free(dp);
  return result;
}

static const char *
status_name(int status)
{
  switch (status) {
    case eslOK:       return "eslOK";
    case eslERANGE:   return "eslERANGE";
    case eslENORESULT:return "eslENORESULT";
    default:          return "other";
  }
}

static const char *
msv_path(int ssv_status, int public_status)
{
  if (ssv_status == eslOK && public_status == eslOK) return "ssv_ok";
  if (ssv_status == eslERANGE && public_status == eslERANGE) return "ssv_erange";
  if (ssv_status == eslENORESULT && public_status == eslOK) return "ssv_fallback_msv_ok";
  if (ssv_status == eslENORESULT && public_status == eslERANGE) return "ssv_fallback_msv_erange";
  return "unexpected";
}

static void
print_json_string(const char *text)
{
  const unsigned char *cursor = (const unsigned char *) (text != NULL ? text : "");
  putchar('"');
  while (*cursor != '\0') {
    unsigned char c = *cursor++;
    switch (c) {
      case '"': fputs("\\\"", stdout); break;
      case '\\': fputs("\\\\", stdout); break;
      case '\b': fputs("\\b", stdout); break;
      case '\f': fputs("\\f", stdout); break;
      case '\n': fputs("\\n", stdout); break;
      case '\r': fputs("\\r", stdout); break;
      case '\t': fputs("\\t", stdout); break;
      default:
        if (c < 0x20) printf("\\u%04x", (unsigned int) c);
        else putchar((int) c);
    }
  }
  putchar('"');
}

static void
print_float_value(float value)
{
  if (isnan(value)) fputs("{\"class\":\"nan\"}", stdout);
  else if (isinf(value)) {
    fputs(signbit(value) ? "{\"class\":\"-inf\"}" : "{\"class\":\"+inf\"}", stdout);
  } else {
    printf("{\"class\":\"finite\",\"hex\":\"%a\"}", (double) value);
  }
}

static void
print_double_value(double value)
{
  if (isnan(value)) fputs("{\"class\":\"nan\"}", stdout);
  else if (isinf(value)) {
    fputs(signbit(value) ? "{\"class\":\"-inf\"}" : "{\"class\":\"+inf\"}", stdout);
  } else {
    printf("{\"class\":\"finite\",\"hex\":\"%a\"}", value);
  }
}

static int
float_bits_equal(float left, float right)
{
  uint32_t left_bits;
  uint32_t right_bits;
  memcpy(&left_bits, &left, sizeof(left_bits));
  memcpy(&right_bits, &right, sizeof(right_bits));
  return left_bits == right_bits;
}

static void
hmm_list_append(HMM_LIST *list, P7_HMM *hmm)
{
  if (list->count == list->capacity) {
    if (list->capacity > SIZE_MAX / 2) fail("too many HMMs");
    size_t new_capacity = list->capacity == 0 ? 4 : list->capacity * 2;
    list->items = xrealloc(list->items,
                           checked_allocation_size(new_capacity, sizeof(*list->items)));
    list->capacity = new_capacity;
  }
  list->items[list->count++] = hmm;
}

static void
sequence_list_append(SEQUENCE_LIST *list, ESL_SQ *sequence)
{
  if (list->count == list->capacity) {
    if (list->capacity > SIZE_MAX / 2) fail("too many sequences");
    size_t new_capacity = list->capacity == 0 ? 128 : list->capacity * 2;
    list->items = xrealloc(list->items,
                           checked_allocation_size(new_capacity, sizeof(*list->items)));
    list->capacity = new_capacity;
  }
  list->items[list->count++] = sequence;
}

static HMM_LIST
read_hmms(const char *path, size_t maximum, ESL_ALPHABET **alphabet)
{
  HMM_LIST list = { NULL, 0, 0 };
  P7_HMMFILE *file = NULL;
  P7_HMM *hmm = NULL;
  char error_buffer[eslERRBUFSIZE];
  int status;

  status = p7_hmmfile_Open(path, NULL, &file, error_buffer);
  if (status != eslOK) fail(error_buffer);
  while ((maximum == 0 || list.count < maximum) &&
         (status = p7_hmmfile_Read(file, alphabet, &hmm)) == eslOK) {
    hmm_list_append(&list, hmm);
    hmm = NULL;
  }
  if (status != eslEOF && !(maximum != 0 && list.count == maximum)) {
    p7_hmmfile_Close(file);
    fail("failed while reading HMM file");
  }
  p7_hmmfile_Close(file);
  if (list.count == 0) fail("HMM input contained no models");
  return list;
}

static SEQUENCE_LIST
read_sequences(const char *path, size_t maximum, ESL_ALPHABET *alphabet)
{
  SEQUENCE_LIST list = { NULL, 0, 0 };
  ESL_SQFILE *file = NULL;
  ESL_SQ *sequence;
  int status;

  status = esl_sqfile_OpenDigital(alphabet, path, eslSQFILE_UNKNOWN, NULL, &file);
  if (status != eslOK) fail("failed to open digital sequence file");
  while (maximum == 0 || list.count < maximum) {
    sequence = esl_sq_CreateDigital(alphabet);
    if (sequence == NULL) fail("failed to allocate digital sequence");
    status = esl_sqio_Read(file, sequence);
    if (status != eslOK) {
      esl_sq_Destroy(sequence);
      break;
    }
    sequence_list_append(&list, sequence);
  }
  esl_sqfile_Close(file);
  if (status != eslEOF && !(maximum != 0 && list.count == maximum))
    fail("failed while reading sequence file");
  if (list.count == 0) fail("sequence input contained no records");
  return list;
}

static void
validate_sequence_lengths(const SEQUENCE_LIST *sequences)
{
  size_t sequence_index;

  for (sequence_index = 0; sequence_index < sequences->count; sequence_index++) {
    const ESL_SQ *sequence = sequences->items[sequence_index];

    if (sequence->n < 0 || sequence->n > HMMER_PIPELINE_MAX_TARGET_LENGTH) {
      fprintf(stderr,
              "msv-oracle: sequence length is outside HMMER's protein pipeline range: "
              "%" PRId64 " (%s)\n",
              sequence->n, sequence->name != NULL ? sequence->name : "");
      exit(EXIT_FAILURE);
    }
  }
}

static size_t
parse_size(const char *text, const char *option)
{
  const char *cursor;
  char *end = NULL;
  unsigned long long value;

  if (text[0] == '\0') goto INVALID;
  for (cursor = text; *cursor != '\0'; cursor++)
    if (*cursor < '0' || *cursor > '9') goto INVALID;

  errno = 0;
  value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value > SIZE_MAX) goto INVALID;
  return (size_t) value;

 INVALID:
  fprintf(stderr, "msv-oracle: invalid %s value: %s\n", option, text);
  exit(EXIT_FAILURE);
}

static double
parse_probability(const char *text)
{
  char *end = NULL;
  double value;
  errno = 0;
  value = strtod(text, &end);
  if (errno != 0 || end == text || *end != '\0' || !isfinite(value) ||
      value < 0.0 || value > 1.0) {
    fprintf(stderr, "msv-oracle: invalid F1 probability: %s\n", text);
    exit(EXIT_FAILURE);
  }
  return value;
}

static void
usage(FILE *stream, const char *program)
{
  fprintf(stream,
          "Usage: %s [--max-models N] [--max-seqs N] [--F1 P] [--strict] HMM FASTA\n",
          program);
}

int
main(int argc, char **argv)
{
  static const struct option options[] = {
    { "max-models", required_argument, NULL, 'm' },
    { "max-seqs",   required_argument, NULL, 's' },
    { "F1",         required_argument, NULL, 'F' },
    { "strict",     no_argument,       NULL, 'x' },
    { "help",       no_argument,       NULL, 'h' },
    { NULL,           0,                 NULL,  0  }
  };
  size_t max_models = 1;
  size_t max_sequences = 100;
  double f1 = 0.02;
  int strict = 0;
  int option;
  ESL_ALPHABET *alphabet = NULL;
  HMM_LIST hmms;
  SEQUENCE_LIST sequences;
  P7_BG *background;
  size_t model_index;
  size_t sequence_index;
  size_t comparisons = 0;
  size_t skipped_empty = 0;
  size_t mismatches = 0;

  while ((option = getopt_long(argc, argv, "m:s:F:xh", options, NULL)) != -1) {
    switch (option) {
      case 'm': max_models = parse_size(optarg, "--max-models"); break;
      case 's': max_sequences = parse_size(optarg, "--max-seqs"); break;
      case 'F': f1 = parse_probability(optarg); break;
      case 'x': strict = 1; break;
      case 'h': usage(stdout, argv[0]); return EXIT_SUCCESS;
      default: usage(stderr, argv[0]); return EXIT_FAILURE;
    }
  }
  if (argc - optind != 2) {
    usage(stderr, argv[0]);
    return EXIT_FAILURE;
  }

  hmms = read_hmms(argv[optind], max_models, &alphabet);
  if (alphabet == NULL || alphabet->type != eslAMINO) fail("only protein HMMs are supported");
  sequences = read_sequences(argv[optind + 1], max_sequences, alphabet);
  validate_sequence_lengths(&sequences);
  background = p7_bg_Create(alphabet);
  if (background == NULL) fail("failed to create HMMER background model");

  fputs("{\"record\":\"metadata\",\"schema_version\":", stdout);
  printf("%d,\"hmmer_revision\":\"%s\",\"hmm_path\":", ORACLE_SCHEMA_VERSION, HMMER_REVISION);
  print_json_string(argv[optind]);
  fputs(",\"fasta_path\":", stdout);
  print_json_string(argv[optind + 1]);
  printf(",\"models_loaded\":%zu,\"sequences_loaded\":%zu,\"F1\":", hmms.count, sequences.count);
  print_double_value(f1);
  fputs("}\n", stdout);

  for (model_index = 0; model_index < hmms.count; model_index++) {
    P7_HMM *hmm = hmms.items[model_index];
    P7_PROFILE *profile;
    P7_OPROFILE *optimized;
    P7_OMX *matrix;
    uint8_t *emissions;
    size_t emission_count;

    if (hmm->M < 1 || alphabet->Kp < 1) fail("invalid model or alphabet dimensions");
    profile = p7_profile_Create(hmm->M, alphabet);
    optimized = p7_oprofile_Create(hmm->M, alphabet);
    matrix = p7_omx_Create(hmm->M, 0, 0);
    if (profile == NULL || optimized == NULL || matrix == NULL) fail("profile allocation failed");
    emission_count = checked_allocation_size((size_t) hmm->M + 1,
                                             (size_t) alphabet->Kp);
    if (p7_ProfileConfig(hmm, background, profile, 100, p7_LOCAL) != eslOK)
      fail("profile configuration failed");
    if (p7_oprofile_Convert(profile, optimized) != eslOK)
      fail("optimized-profile conversion failed");
    emissions = malloc(emission_count * sizeof(*emissions));
    if (emissions == NULL) fail("emission-array allocation failed");
    if (p7_oprofile_GetSSVEmissionScoreArray(optimized, emissions) != eslOK)
      fail("emission extraction failed");

    for (sequence_index = 0; sequence_index < sequences.count; sequence_index++) {
      ESL_SQ *sequence = sequences.items[sequence_index];
      SCALAR_MSV_RESULT scalar;
      float ssv_score = NAN;
      float public_score = NAN;
      float null_score;
      float bit_score;
      double probability;
      int ssv_status;
      int public_status;
      int status_agreement;
      int score_agreement;
      int pass;
      int length;

      length = (int) sequence->n;

      /* p7_Pipeline() returns before null1/MSV for L=0. Calling the filter
       * API directly at L=0 produces an SSV value outside its supported
       * pipeline preconditions, so count the pair as not reached. */
      if (length == 0) {
        skipped_empty++;
        continue;
      }

      if (p7_bg_SetLength(background, length) != eslOK ||
          p7_oprofile_ReconfigLength(optimized, length) != eslOK ||
          p7_omx_GrowTo(matrix, optimized->M, 0, length) != eslOK)
        fail("per-sequence reconfiguration failed");
      if (p7_bg_NullOne(background, sequence->dsq, length, &null_score) != eslOK)
        fail("null1 scoring failed");

      ssv_status = p7_SSVFilter(sequence->dsq, length, optimized, &ssv_score);
      public_status = p7_MSVFilter(sequence->dsq, length, optimized, matrix, &public_score);
      scalar = scalar_full_msv(sequence->dsq, length, optimized, emissions);

      status_agreement = (public_status == scalar.status);
      score_agreement = float_bits_equal(public_score, scalar.score);
      if (!status_agreement || !score_agreement) mismatches++;
      comparisons++;

      bit_score = (float) ((public_score - null_score) / eslCONST_LOG2);
      probability = esl_gumbel_surv(bit_score,
                                    optimized->evparam[p7_MMU],
                                    optimized->evparam[p7_MLAMBDA]);
      pass = probability <= f1;

      fputs("{\"record\":\"comparison\",\"schema_version\":", stdout);
      printf("%d,\"model_index\":%zu,\"sequence_index\":%zu,\"model_name\":",
             ORACLE_SCHEMA_VERSION, model_index, sequence_index);
      print_json_string(hmm->name);
      fputs(",\"sequence_name\":", stdout);
      print_json_string(sequence->name);
      printf(",\"M\":%d,\"L\":%" PRId64, hmm->M, sequence->n);
      printf(",\"profile_u8\":{\"tbm_b\":%u,\"tec_b\":%u,\"tjb_b\":%u,\"base_b\":%u,\"bias_b\":%u,\"scale_b\":",
             (unsigned int) optimized->tbm_b, (unsigned int) optimized->tec_b,
             (unsigned int) optimized->tjb_b, (unsigned int) optimized->base_b,
             (unsigned int) optimized->bias_b);
      print_float_value(optimized->scale_b);
      printf("},\"msv_path\":\"%s\",\"ssv\":{\"status\":\"%s\",\"score\":",
             msv_path(ssv_status, public_status), status_name(ssv_status));
      print_float_value(ssv_score);
      printf("},\"public_msv\":{\"status\":\"%s\",\"score\":", status_name(public_status));
      print_float_value(public_score);
      printf("},\"scalar_full_msv\":{\"status\":\"%s\",\"xJ_u8\":%u,\"overflow_row\":%d,\"overflow_xE_u8\":%u,\"score\":",
             status_name(scalar.status), (unsigned int) scalar.xj, scalar.overflow_row,
             (unsigned int) scalar.overflow_xe);
      print_float_value(scalar.score);
      fputs("},\"null_score\":", stdout);
      print_float_value(null_score);
      fputs(",\"bit_score\":", stdout);
      print_float_value(bit_score);
      fputs(",\"P\":", stdout);
      print_double_value(probability);
      printf(",\"pass_F1\":%s,\"agreement\":{\"status\":%s,\"score_bits\":%s}}\n",
             pass ? "true" : "false", status_agreement ? "true" : "false",
             score_agreement ? "true" : "false");
      p7_omx_Reuse(matrix);
    }

    free(emissions);
    p7_omx_Destroy(matrix);
    p7_oprofile_Destroy(optimized);
    p7_profile_Destroy(profile);
  }

  printf("{\"record\":\"summary\",\"schema_version\":%d,\"comparisons\":%zu,"
         "\"skipped_empty\":%zu,\"mismatches\":%zu}\n",
         ORACLE_SCHEMA_VERSION, comparisons, skipped_empty, mismatches);

  p7_bg_Destroy(background);
  for (sequence_index = 0; sequence_index < sequences.count; sequence_index++)
    esl_sq_Destroy(sequences.items[sequence_index]);
  free(sequences.items);
  for (model_index = 0; model_index < hmms.count; model_index++)
    p7_hmm_Destroy(hmms.items[model_index]);
  free(hmms.items);
  esl_alphabet_Destroy(alphabet);

  if (strict && mismatches != 0) return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
