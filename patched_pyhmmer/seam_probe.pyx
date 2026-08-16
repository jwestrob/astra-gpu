# cython: language_level=3

from libc.stddef cimport size_t
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcmp, memcpy, memset
from libc.math cimport exp, fabsf, log, logf

from libeasel cimport (
    ESL_DSQ,
    eslCONST_LOG2,
    eslEINACCURATE,
    eslEINVAL,
    eslEMEM,
    eslENORESULT,
    eslERANGE,
    eslOK,
)
from libeasel.sq cimport ESL_SQ
from libeasel.alphabet cimport (
    ESL_ALPHABET,
    eslAMINO,
    eslDNA,
    esl_alphabet_Create,
    esl_alphabet_CreateCustom,
    esl_alphabet_Destroy,
)
from libhmmer cimport p7_FLAMBDA, p7_FTAU
from libhmmer.impl_sse cimport p7_MSVFilter
from libhmmer.impl.p7_omx cimport P7_OMX, p7_omx_GrowTo
from libhmmer.impl.p7_oprofile cimport (
    P7_OPROFILE,
    p7_oprofile_ReconfigLength,
    p7_oprofile_ReconfigMultihit,
    p7_oprofile_ReconfigUnihit,
)
from libhmmer.p7_domaindef cimport P7_DOMAINDEF, p7_domaindef_GrowTo
from libhmmer.p7_domain cimport P7_DOMAIN
from libhmmer.p7_bg cimport (
    P7_BG,
    p7_bg_FilterScore,
    p7_bg_NullOne,
    p7_bg_SetLength,
)
from libhmmer.logsum cimport p7_FLogsum
from libhmmer.p7_pipeline cimport (
    P7_PIPELINE,
    P7_PIPELINE_COMPACT_DOMAIN,
    P7_PIPELINE_COMPACT_TRACE_STEP,
    P7_PIPELINE_SIMPLE_REGION,
    p7_COMPACT_DOMAIN_DEVICE_RESULT,
    p7_COMPACT_NULL2_COUNT,
    p7_DOMAIN_NO_REGIONS,
    p7_DOMAIN_SIMPLE,
    p7_SEARCH_SEQS,
    p7_ZSETBY_NTARGETS,
    p7_ZSETBY_OPTION,
    p7_VIT_CPU,
    p7_VIT_EXTERNAL,
    p7_VIT_NONE,
    p7_Pipeline,
    p7_PipelineFromMSV,
    p7_PipelineFromFilterScores,
    p7_PipelineFromFilterAndForwardScores,
    p7_PipelineFromFilterAndForwardSimpleRegions,
    p7_PipelineFromFilterForwardAndCompactDomainsV2,
    p7_pipeline_CompactTailFingerprintV2,
    p7_pipeline_Reuse,
    p7_pipeline_domain_route_e,
    p7_pipeline_vitmode_e,
    p7_pli_NewModel,
    p7_pli_NewSeq,
)
from libhmmer.p7_trace cimport (
    P7_TRACE,
    p7T_M,
    p7_trace_Reuse,
)
from libhmmer.p7_tophits cimport P7_TOPHITS

from pyhmmer.easel cimport DigitalSequenceBlock
from pyhmmer.plan7 cimport HMM, OptimizedProfile, Pipeline, TopHits


cdef extern from "hmmer.h" nogil:
    int p7_compact_AlphabetIsCanonicalAminoV2(const ESL_ALPHABET *abc)
    int p7_compact_AlphabetsAreEqualV2(
        const ESL_ALPHABET *left,
        const ESL_ALPHABET *right,
    )
    int p7_compact_AlphabetSetIsExactV2(
        const ESL_ALPHABET *profile,
        const ESL_ALPHABET *background,
        const ESL_ALPHABET *filter,
        const ESL_ALPHABET *sequence,
    )


cdef extern from "fenv.h" nogil:
    int FE_DOWNWARD
    int fegetround()
    int fesetround(int round)


cdef extern from "esl_exponential.h" nogil:
    double esl_exp_logsurv(double x, double mu, double lambda_)


cdef enum:
    GPU_VITERBI_NOT_RUN_C = -1
    FORWARD_SPECIAL_CELLS = 6

GPU_VITERBI_NOT_RUN = GPU_VITERBI_NOT_RUN_C
GPU_VITERBI_OK = eslOK
GPU_VITERBI_ERANGE = eslERANGE
GPU_VITERBI_ENORESULT = eslENORESULT
HMMER_EINVAL = eslEINVAL
HMMER_EINACCURATE = eslEINACCURATE


cdef extern from "impl_sse/impl_sse.h" nogil:
    int p7_ViterbiFilter(
        const unsigned char *dsq,
        int L,
        const P7_OPROFILE *om,
        P7_OMX *ox,
        float *ret_sc,
    )
    int p7_ForwardParser(
        const unsigned char *dsq,
        int L,
        const P7_OPROFILE *om,
        P7_OMX *ox,
        float *ret_sc,
    )
    int p7_BackwardParser(
        const unsigned char *dsq,
        int L,
        const P7_OPROFILE *om,
        const P7_OMX *fwd,
        P7_OMX *bck,
        float *ret_sc,
    )
    int p7_DomainDecoding(
        const P7_OPROFILE *om,
        const P7_OMX *fwd,
        const P7_OMX *bck,
        P7_DOMAINDEF *ddef,
    )
    int p7_Forward(
        const unsigned char *dsq,
        int L,
        const P7_OPROFILE *om,
        P7_OMX *ox,
        float *ret_sc,
    )
    int p7_Backward(
        const unsigned char *dsq,
        int L,
        const P7_OPROFILE *om,
        const P7_OMX *fwd,
        P7_OMX *bck,
        float *ret_sc,
    )
    int p7_Decoding(
        const P7_OPROFILE *om,
        const P7_OMX *fwd,
        P7_OMX *bck,
        P7_OMX *pp,
    )
    int p7_OptimalAccuracy(
        const P7_OPROFILE *om,
        const P7_OMX *pp,
        P7_OMX *ox,
        float *ret_e,
    )
    int p7_OATrace(
        const P7_OPROFILE *om,
        const P7_OMX *pp,
        const P7_OMX *ox,
        P7_TRACE *tr,
    )
    int p7_Null2_ByExpectation(
        const P7_OPROFILE *om,
        const P7_OMX *pp,
        float *null2,
    )


cdef union _float_bits:
    float value
    uint32_t bits


cdef union _double_bits:
    double value
    uint64_t bits


cdef int _loop_msv(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    P7_TOPHITS* th,
) noexcept nogil:
    cdef float usc
    cdef int status
    cdef size_t t

    status = p7_pli_NewModel(pli, om, bg)
    if status != eslOK:
        return status
    for t in range(n_targets):
        status = p7_pli_NewSeq(pli, sq[t])
        if status != eslOK:
            return status
        status = p7_bg_SetLength(bg, sq[t].n)
        if status != eslOK:
            return status
        status = p7_oprofile_ReconfigLength(om, sq[t].n)
        if status != eslOK:
            return status
        status = p7_omx_GrowTo(pli.oxf, om.M, 0, sq[t].n)
        if status != eslOK:
            return status
        p7_MSVFilter(sq[t].dsq, sq[t].n, om, pli.oxf, &usc)
        status = p7_PipelineFromMSV(pli, om, bg, sq[t], NULL, th, usc)
        if status != eslOK:
            return status
        status = p7_pipeline_Reuse(pli)
        if status != eslOK:
            return status
    return eslOK


cdef int _loop(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    P7_TOPHITS* th,
    int gpu_viterbi_status,
    bint poison_filtersc,
) noexcept nogil:
    cdef float filtersc
    cdef float usc
    cdef float vfsc
    cdef p7_pipeline_vitmode_e vitmode
    cdef int status
    cdef int vitstatus
    cdef size_t t

    status = p7_pli_NewModel(pli, om, bg)
    if status == eslOK:
        for t in range(n_targets):
            status = p7_pli_NewSeq(pli, sq[t])
            if status != eslOK:
                break
            status = p7_bg_SetLength(bg, sq[t].n)
            if status != eslOK:
                break
            status = p7_oprofile_ReconfigLength(om, sq[t].n)
            if status != eslOK:
                break
            status = p7_omx_GrowTo(pli.oxf, om.M, 0, sq[t].n)
            if status != eslOK:
                break

            p7_MSVFilter(sq[t].dsq, sq[t].n, om, pli.oxf, &usc)
            status = p7_bg_FilterScore(
                bg, sq[t].dsq, sq[t].n, &filtersc,
            )
            if status != eslOK:
                break
            if poison_filtersc:
                filtersc = -1.0e30

            if gpu_viterbi_status == eslOK:
                vitstatus = p7_ViterbiFilter(
                    sq[t].dsq, sq[t].n, om, pli.oxf, &vfsc,
                )
                vitmode = (
                    p7_VIT_EXTERNAL if vitstatus == eslOK else p7_VIT_CPU
                )
            elif (
                gpu_viterbi_status == eslERANGE
                or gpu_viterbi_status == eslENORESULT
            ):
                vitmode = p7_VIT_CPU
                vfsc = 0.0
            elif gpu_viterbi_status == GPU_VITERBI_NOT_RUN_C:
                vitmode = p7_VIT_NONE
                vfsc = 0.0
            else:
                status = eslEINVAL
                break

            status = p7_PipelineFromFilterScores(
                pli, om, bg, sq[t], NULL, th,
                usc, filtersc, vitmode, vfsc,
            )
            if status != eslOK:
                break
            status = p7_pipeline_Reuse(pli)
            if status != eslOK:
                break

    return status


cdef int _loop_forward(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    P7_TOPHITS* th,
    bint omit_forward_matrix,
) noexcept nogil:
    cdef float* forward_xmx = NULL
    cdef uint64_t xmx_count
    cdef float filtersc
    cdef float fwdsc
    cdef float usc
    cdef float vfsc
    cdef int status
    cdef size_t t

    status = p7_pli_NewModel(pli, om, bg)
    if status == eslOK:
        for t in range(n_targets):
            status = p7_pli_NewSeq(pli, sq[t])
            if status != eslOK:
                break
            status = p7_bg_SetLength(bg, sq[t].n)
            if status != eslOK:
                break
            status = p7_oprofile_ReconfigLength(om, sq[t].n)
            if status != eslOK:
                break
            status = p7_omx_GrowTo(pli.oxf, om.M, 0, sq[t].n)
            if status != eslOK:
                break

            status = p7_MSVFilter(sq[t].dsq, sq[t].n, om, pli.oxf, &usc)
            if status != eslOK:
                break
            status = p7_bg_FilterScore(bg, sq[t].dsq, sq[t].n, &filtersc)
            if status != eslOK:
                break
            status = p7_ViterbiFilter(
                sq[t].dsq, sq[t].n, om, pli.oxf, &vfsc,
            )
            if status != eslOK:
                break
            status = p7_ForwardParser(
                sq[t].dsq, sq[t].n, om, pli.oxf, &fwdsc,
            )
            if status != eslOK:
                break

            xmx_count = <uint64_t> (sq[t].n + 1) * FORWARD_SPECIAL_CELLS
            if not omit_forward_matrix:
                forward_xmx = <float*> malloc(xmx_count * sizeof(float))
                if forward_xmx == NULL:
                    status = eslEMEM
                    break
                memcpy(forward_xmx, pli.oxf.xmx, xmx_count * sizeof(float))

            status = p7_PipelineFromFilterAndForwardScores(
                pli, om, bg, sq[t], NULL, th,
                usc, filtersc, vfsc, fwdsc,
                forward_xmx,
                0 if omit_forward_matrix else xmx_count,
            )
            free(forward_xmx)
            forward_xmx = NULL
            if status != eslOK:
                break
            status = p7_pipeline_Reuse(pli)
            if status != eslOK:
                break

    free(forward_xmx)
    return status


cdef int _extract_domain_route(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    const ESL_SQ* sq,
    P7_PIPELINE_SIMPLE_REGION** ret_regions,
    uint64_t* ret_region_count,
    float* ret_nexpected,
    bint* ret_clustered,
) noexcept nogil:
    cdef P7_PIPELINE_SIMPLE_REGION* regions = NULL
    cdef float bdelta
    cdef float edelta
    cdef float expected
    cdef float left
    cdef float max_expected
    cdef float right
    cdef int i = -1
    cdef int j
    cdef int status
    cdef int z
    cdef uint64_t count = 0
    cdef bint triggered = False

    ret_regions[0] = NULL
    ret_region_count[0] = 0
    ret_nexpected[0] = 0.0
    ret_clustered[0] = False
    if sq.n <= 0:
        return eslEINVAL

    status = p7_omx_GrowTo(pli.oxb, om.M, 0, sq.n)
    if status != eslOK:
        return status
    status = p7_BackwardParser(
        sq.dsq, sq.n, om, pli.oxf, pli.oxb, NULL,
    )
    if status != eslOK:
        return status
    status = p7_domaindef_GrowTo(pli.ddef, sq.n)
    if status != eslOK:
        return status
    status = p7_DomainDecoding(om, pli.oxf, pli.oxb, pli.ddef)
    if status != eslOK:
        return status

    regions = <P7_PIPELINE_SIMPLE_REGION*> malloc(
        <size_t> sq.n * sizeof(P7_PIPELINE_SIMPLE_REGION)
    )
    if regions == NULL:
        return eslEMEM

    for j in range(1, sq.n + 1):
        bdelta = pli.ddef.btot[j] - pli.ddef.btot[j - 1]
        edelta = pli.ddef.etot[j] - pli.ddef.etot[j - 1]
        if not triggered:
            if pli.ddef.mocc[j] - bdelta < pli.ddef.rt2:
                i = j
            elif i == -1:
                i = j
            if pli.ddef.mocc[j] >= pli.ddef.rt1:
                triggered = True
        elif pli.ddef.mocc[j] - edelta < pli.ddef.rt2:
            regions[count].i = <uint32_t> i
            regions[count].j = <uint32_t> j
            count += 1
            max_expected = -1.0
            for z in range(i, j + 1):
                left = pli.ddef.etot[z] - pli.ddef.etot[i - 1]
                right = pli.ddef.btot[j] - pli.ddef.btot[z - 1]
                expected = left if left < right else right
                if expected > max_expected:
                    max_expected = expected
            if max_expected >= pli.ddef.rt3:
                ret_clustered[0] = True
            i = -1
            triggered = False

    if triggered:
        free(regions)
        return eslEINVAL
    if count == 0:
        free(regions)
        regions = NULL
    ret_regions[0] = regions
    ret_region_count[0] = count
    ret_nexpected[0] = pli.ddef.btot[sq.n]
    return eslOK


cdef int _build_compact_domains(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    const ESL_SQ* sq,
    const P7_PIPELINE_SIMPLE_REGION* regions,
    uint64_t region_count,
    uint32_t row_index,
    uint32_t profile_index,
    uint32_t sequence_index,
    P7_PIPELINE_COMPACT_DOMAIN* domains,
    float* null2,
    uint64_t* trace_offsets,
    P7_PIPELINE_COMPACT_TRACE_STEP* traces,
    uint64_t trace_capacity,
    uint64_t* ret_trace_count,
) noexcept nogil:
    """Build a pristine CPU oracle payload for the compact seam tests."""
    cdef P7_PIPELINE_COMPACT_DOMAIN* domain
    cdef P7_PIPELINE_COMPACT_TRACE_STEP* step
    cdef P7_TRACE* trace = pli.ddef.tr
    cdef float* domain_null2
    cdef float backward_score
    cdef float correction
    cdef float forward_score
    cdef float oa_score
    cdef uint64_t trace_used = 0
    cdef uint64_t d
    cdef int first_match
    cdef int first_model
    cdef int last_match
    cdef int last_model
    cdef int restore_status
    cdef int status
    cdef int i
    cdef int j
    cdef int L
    cdef int pos
    cdef int z

    ret_trace_count[0] = 0
    if (
        pli == NULL or om == NULL or sq == NULL or trace == NULL
        or regions == NULL or region_count == 0 or domains == NULL
        or null2 == NULL or trace_offsets == NULL or traces == NULL
    ):
        return eslEINVAL
    status = p7_oprofile_ReconfigUnihit(om, sq.n)
    if status != eslOK:
        return status
    for d in range(region_count):
        i = <int> regions[d].i
        j = <int> regions[d].j
        L = j - i + 1
        if i <= 0 or L <= 0 or j > sq.n:
            status = eslEINVAL
            break
        status = p7_omx_GrowTo(pli.fwd, om.M, L, L)
        if status != eslOK:
            break
        status = p7_omx_GrowTo(pli.bck, om.M, L, L)
        if status != eslOK:
            break
        status = p7_Forward(
            sq.dsq + i - 1, L, om, pli.fwd, &forward_score,
        )
        if status != eslOK:
            break
        status = p7_Backward(
            sq.dsq + i - 1, L, om, pli.fwd, pli.bck,
            &backward_score,
        )
        if status != eslOK:
            break
        status = p7_Decoding(om, pli.fwd, pli.bck, pli.bck)
        if status != eslOK:
            break
        status = p7_OptimalAccuracy(om, pli.bck, pli.fwd, &oa_score)
        if status != eslOK:
            break
        status = p7_OATrace(om, pli.bck, pli.fwd, trace)
        if status != eslOK:
            break
        domain_null2 = null2 + d * p7_COMPACT_NULL2_COUNT
        status = p7_Null2_ByExpectation(om, pli.bck, domain_null2)
        if status != eslOK:
            break
        if (
            trace.N < 0 or <uint64_t> trace.N > trace_capacity
            or trace_used > trace_capacity - <uint64_t> trace.N
        ):
            status = eslERANGE
            break

        domain = domains + d
        memset(domain, 0, sizeof(P7_PIPELINE_COMPACT_DOMAIN))
        domain.row_index = row_index
        domain.profile_index = profile_index
        domain.sequence_index = sequence_index
        domain.envelope_begin = <uint32_t> i
        domain.envelope_end = <uint32_t> j
        domain.forward_score = forward_score
        domain.backward_score = backward_score
        domain.oa_score = oa_score
        domain.score_consistency = fabsf(forward_score - backward_score)
        domain.status = <uint8_t> eslOK
        domain.action = <uint8_t> p7_COMPACT_DOMAIN_DEVICE_RESULT
        domain.has_own_scales = <uint8_t> pli.bck.has_own_scales

        trace_offsets[d] = trace_used
        first_match = 0
        first_model = 0
        last_match = 0
        last_model = 0
        for z in range(trace.N):
            step = traces + trace_used + <uint64_t> z
            memset(step, 0, sizeof(P7_PIPELINE_COMPACT_TRACE_STEP))
            step.state = <uint8_t> trace.st[z]
            step.model_position = <uint32_t> trace.k[z]
            if trace.i[z] > 0:
                step.sequence_position = <uint32_t> (trace.i[z] + i - 1)
            step.posterior = trace.pp[z]
            if trace.st[z] == p7T_M:
                if first_match == 0:
                    first_match = <int> step.sequence_position
                    first_model = <int> step.model_position
                last_match = <int> step.sequence_position
                last_model = <int> step.model_position
        if first_match == 0:
            status = eslENORESULT
            break
        trace_used += <uint64_t> trace.N
        domain.alignment_begin = <uint32_t> first_match
        domain.alignment_end = <uint32_t> last_match
        domain.model_begin = <uint32_t> first_model
        domain.model_end = <uint32_t> last_model
        correction = 0.0
        for pos in range(i, j + 1):
            correction += logf(domain_null2[sq.dsq[pos]])
        domain.domain_correction = correction
        p7_trace_Reuse(trace)

    if status == eslOK:
        trace_offsets[region_count] = trace_used
        ret_trace_count[0] = trace_used
    p7_trace_Reuse(trace)
    restore_status = p7_oprofile_ReconfigMultihit(om, sq.n)
    if status == eslOK:
        status = restore_status
    return status


cdef int _loop_simple_regions(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    P7_TOPHITS* th,
    uint64_t* route_counts,
) noexcept nogil:
    cdef P7_PIPELINE_SIMPLE_REGION* regions = NULL
    cdef float* forward_xmx = NULL
    cdef _double_bits f1_bits
    cdef _double_bits f2_bits
    cdef _double_bits f3_bits
    cdef uint64_t region_count
    cdef uint64_t xmx_count
    cdef float filtersc
    cdef float fwdsc
    cdef float nexpected
    cdef float usc
    cdef float vfsc
    cdef p7_pipeline_domain_route_e route
    cdef int status
    cdef size_t t
    cdef bint clustered

    route_counts[0] = 0
    route_counts[1] = 0
    route_counts[2] = 0
    status = p7_pli_NewModel(pli, om, bg)
    if status == eslOK:
        for t in range(n_targets):
            status = p7_pli_NewSeq(pli, sq[t])
            if status != eslOK:
                break
            status = p7_bg_SetLength(bg, sq[t].n)
            if status != eslOK:
                break
            status = p7_oprofile_ReconfigLength(om, sq[t].n)
            if status != eslOK:
                break
            status = p7_omx_GrowTo(pli.oxf, om.M, 0, sq[t].n)
            if status != eslOK:
                break
            status = p7_MSVFilter(sq[t].dsq, sq[t].n, om, pli.oxf, &usc)
            if status != eslOK:
                break
            status = p7_bg_FilterScore(bg, sq[t].dsq, sq[t].n, &filtersc)
            if status != eslOK:
                break
            status = p7_ViterbiFilter(
                sq[t].dsq, sq[t].n, om, pli.oxf, &vfsc,
            )
            if status != eslOK:
                break
            status = p7_ForwardParser(
                sq[t].dsq, sq[t].n, om, pli.oxf, &fwdsc,
            )
            if status != eslOK:
                break
            status = _extract_domain_route(
                pli, om, sq[t], &regions, &region_count,
                &nexpected, &clustered,
            )
            if status != eslOK:
                break

            if clustered:
                xmx_count = <uint64_t> (sq[t].n + 1) * FORWARD_SPECIAL_CELLS
                forward_xmx = <float*> malloc(xmx_count * sizeof(float))
                if forward_xmx == NULL:
                    status = eslEMEM
                    break
                memcpy(forward_xmx, pli.oxf.xmx, xmx_count * sizeof(float))
                status = p7_PipelineFromFilterAndForwardScores(
                    pli, om, bg, sq[t], NULL, th,
                    usc, filtersc, vfsc, fwdsc, forward_xmx, xmx_count,
                )
                route_counts[0] += 1
            else:
                f1_bits.value = pli.F1
                f2_bits.value = pli.F2
                f3_bits.value = pli.F3
                route = (
                    p7_DOMAIN_NO_REGIONS
                    if region_count == 0
                    else p7_DOMAIN_SIMPLE
                )
                status = p7_PipelineFromFilterAndForwardSimpleRegions(
                    pli, om, bg, sq[t], NULL, th,
                    usc, filtersc, vfsc, fwdsc,
                    f1_bits.bits, f2_bits.bits, f3_bits.bits,
                    pli.do_biasfilter, route, nexpected,
                    regions, region_count,
                )
                route_counts[<size_t> route] += 1
            free(forward_xmx)
            forward_xmx = NULL
            free(regions)
            regions = NULL
            if status != eslOK:
                break
            status = p7_pipeline_Reuse(pli)
            if status != eslOK:
                break

    free(forward_xmx)
    free(regions)
    return status


cdef int _loop_compact_domains(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    P7_TOPHITS* th,
    uint64_t* route_counts,
    float compact_fwd_delta,
) noexcept nogil:
    cdef P7_PIPELINE_SIMPLE_REGION* regions = NULL
    cdef P7_PIPELINE_COMPACT_DOMAIN* domains = NULL
    cdef P7_PIPELINE_COMPACT_TRACE_STEP* traces = NULL
    cdef uint64_t* trace_offsets = NULL
    cdef float* null2 = NULL
    cdef float* forward_xmx = NULL
    cdef _double_bits f1_bits
    cdef _double_bits f2_bits
    cdef _double_bits f3_bits
    cdef uint64_t generation_tail_fingerprint
    cdef uint64_t region_count
    cdef uint64_t trace_capacity
    cdef uint64_t trace_count
    cdef uint64_t xmx_count
    cdef float filtersc
    cdef float fwdsc
    cdef float nexpected
    cdef float usc
    cdef float vfsc
    cdef p7_pipeline_domain_route_e route
    cdef int status
    cdef size_t t
    cdef bint clustered

    route_counts[0] = 0
    route_counts[1] = 0
    route_counts[2] = 0
    route_counts[3] = 0
    route_counts[4] = 0
    status = p7_pli_NewModel(pli, om, bg)
    if status == eslOK:
        for t in range(n_targets):
            status = p7_pli_NewSeq(pli, sq[t])
            if status != eslOK:
                break
            status = p7_bg_SetLength(bg, sq[t].n)
            if status != eslOK:
                break
            status = p7_oprofile_ReconfigLength(om, sq[t].n)
            if status != eslOK:
                break
            status = p7_omx_GrowTo(pli.oxf, om.M, 0, sq[t].n)
            if status != eslOK:
                break
            status = p7_MSVFilter(sq[t].dsq, sq[t].n, om, pli.oxf, &usc)
            if status != eslOK:
                break
            status = p7_bg_FilterScore(bg, sq[t].dsq, sq[t].n, &filtersc)
            if status != eslOK:
                break
            status = p7_ViterbiFilter(
                sq[t].dsq, sq[t].n, om, pli.oxf, &vfsc,
            )
            if status != eslOK:
                break
            status = p7_ForwardParser(
                sq[t].dsq, sq[t].n, om, pli.oxf, &fwdsc,
            )
            if status != eslOK:
                break
            status = _extract_domain_route(
                pli, om, sq[t], &regions, &region_count,
                &nexpected, &clustered,
            )
            if status != eslOK:
                break

            if clustered:
                xmx_count = <uint64_t> (sq[t].n + 1) * FORWARD_SPECIAL_CELLS
                forward_xmx = <float*> malloc(xmx_count * sizeof(float))
                if forward_xmx == NULL:
                    status = eslEMEM
                    break
                memcpy(forward_xmx, pli.oxf.xmx, xmx_count * sizeof(float))
                status = p7_PipelineFromFilterAndForwardScores(
                    pli, om, bg, sq[t], NULL, th,
                    usc, filtersc, vfsc, fwdsc, forward_xmx, xmx_count,
                )
                route_counts[0] += 1
            elif region_count == 0:
                f1_bits.value = pli.F1
                f2_bits.value = pli.F2
                f3_bits.value = pli.F3
                route = p7_DOMAIN_NO_REGIONS
                status = p7_PipelineFromFilterAndForwardSimpleRegions(
                    pli, om, bg, sq[t], NULL, th,
                    usc, filtersc, vfsc, fwdsc,
                    f1_bits.bits, f2_bits.bits, f3_bits.bits,
                    pli.do_biasfilter, route, nexpected, NULL, 0,
                )
                route_counts[1] += 1
            else:
                xmx_count = <uint64_t> (sq[t].n + 1) * FORWARD_SPECIAL_CELLS
                forward_xmx = <float*> malloc(xmx_count * sizeof(float))
                trace_capacity = (
                    <uint64_t> sq[t].n
                    + region_count * (<uint64_t> om.M + 6)
                )
                domains = <P7_PIPELINE_COMPACT_DOMAIN*> malloc(
                    region_count * sizeof(P7_PIPELINE_COMPACT_DOMAIN)
                )
                null2 = <float*> malloc(
                    region_count * p7_COMPACT_NULL2_COUNT * sizeof(float)
                )
                trace_offsets = <uint64_t*> malloc(
                    (region_count + 1) * sizeof(uint64_t)
                )
                traces = <P7_PIPELINE_COMPACT_TRACE_STEP*> malloc(
                    trace_capacity * sizeof(P7_PIPELINE_COMPACT_TRACE_STEP)
                )
                if (
                    forward_xmx == NULL or domains == NULL or null2 == NULL
                    or trace_offsets == NULL or traces == NULL
                ):
                    status = eslEMEM
                    break
                memcpy(forward_xmx, pli.oxf.xmx, xmx_count * sizeof(float))
                status = _build_compact_domains(
                    pli, om, sq[t], regions, region_count,
                    <uint32_t> t, 0, <uint32_t> t,
                    domains, null2, trace_offsets, traces,
                    trace_capacity, &trace_count,
                )
                if status != eslOK:
                    break
                generation_tail_fingerprint = (
                    p7_pipeline_CompactTailFingerprintV2(pli)
                )
                status = p7_PipelineFromFilterForwardAndCompactDomainsV2(
                    pli, om, bg, sq[t], NULL, th,
                    usc, filtersc, vfsc, fwdsc + compact_fwd_delta,
                    generation_tail_fingerprint,
                    <uint64_t> n_targets,
                    <uint32_t> t, 0, <uint32_t> t, nexpected,
                    domains, region_count,
                    trace_offsets, region_count + 1,
                    traces, trace_count,
                    null2, region_count * p7_COMPACT_NULL2_COUNT,
                )
                if status == eslEINACCURATE:
                    status = p7_Pipeline(
                        pli, om, bg, sq[t], NULL, th,
                    )
                    route_counts[4] += 1
                else:
                    route_counts[3] += 1

            free(forward_xmx)
            forward_xmx = NULL
            free(domains)
            domains = NULL
            free(null2)
            null2 = NULL
            free(trace_offsets)
            trace_offsets = NULL
            free(traces)
            traces = NULL
            free(regions)
            regions = NULL
            if status != eslOK:
                break
            status = p7_pipeline_Reuse(pli)
            if status != eslOK:
                break

    free(forward_xmx)
    free(domains)
    free(null2)
    free(trace_offsets)
    free(traces)
    free(regions)
    return status


cdef int _invalid_forward(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ* sq,
    P7_TOPHITS* th,
    int corruption,
    uint64_t* ret_changed,
) noexcept nogil:
    cdef _float_bits bits
    cdef float* forward_xmx = NULL
    cdef float* original_xmx = NULL
    cdef const float* supplied_xmx
    cdef uint64_t xmx_count
    cdef uint64_t supplied_count
    cdef uint64_t n_past_msv
    cdef uint64_t n_past_bias
    cdef uint64_t n_past_vit
    cdef uint64_t n_past_fwd
    cdef float filtersc
    cdef float fwdsc
    cdef float usc
    cdef float vfsc
    cdef int matrix_changed
    cdef int original_rounding
    cdef int status

    ret_changed[0] = 1
    status = p7_pli_NewModel(pli, om, bg)
    if status != eslOK:
        return status
    status = p7_pli_NewSeq(pli, sq)
    if status != eslOK:
        return status
    status = p7_bg_SetLength(bg, sq.n)
    if status != eslOK:
        return status
    status = p7_oprofile_ReconfigLength(om, sq.n)
    if status != eslOK:
        return status
    status = p7_omx_GrowTo(pli.oxf, om.M, 0, sq.n)
    if status != eslOK:
        return status
    status = p7_MSVFilter(sq.dsq, sq.n, om, pli.oxf, &usc)
    if status != eslOK:
        return status
    status = p7_bg_FilterScore(bg, sq.dsq, sq.n, &filtersc)
    if status != eslOK:
        return status
    status = p7_ViterbiFilter(sq.dsq, sq.n, om, pli.oxf, &vfsc)
    if status != eslOK:
        return status
    status = p7_ForwardParser(sq.dsq, sq.n, om, pli.oxf, &fwdsc)
    if status != eslOK:
        return status

    xmx_count = <uint64_t> (sq.n + 1) * FORWARD_SPECIAL_CELLS
    supplied_count = xmx_count
    forward_xmx = <float*> malloc(xmx_count * sizeof(float))
    original_xmx = <float*> malloc(xmx_count * sizeof(float))
    if forward_xmx == NULL or original_xmx == NULL:
        free(original_xmx)
        free(forward_xmx)
        return eslEMEM
    memcpy(forward_xmx, pli.oxf.xmx, xmx_count * sizeof(float))
    memcpy(original_xmx, pli.oxf.xmx, xmx_count * sizeof(float))
    supplied_xmx = forward_xmx

    if corruption == 0:
        supplied_count -= 1
    elif corruption == 1:
        bits.value = forward_xmx[FORWARD_SPECIAL_CELLS + 1]
        bits.bits ^= 1
        forward_xmx[FORWARD_SPECIAL_CELLS + 1] = bits.value
    elif corruption == 2:
        bits.value = fwdsc
        bits.bits ^= 1
        fwdsc = bits.value
    elif corruption == 3:
        forward_xmx[FORWARD_SPECIAL_CELLS + 5] = 2.0
    elif corruption == 4:
        supplied_xmx = NULL
        supplied_count = 0
    elif corruption == 5:
        original_rounding = fegetround()
        if fesetround(FE_DOWNWARD) != 0:
            free(original_xmx)
            free(forward_xmx)
            return eslEINVAL
    else:
        free(original_xmx)
        free(forward_xmx)
        return eslEINVAL

    n_past_msv = pli.n_past_msv
    n_past_bias = pli.n_past_bias
    n_past_vit = pli.n_past_vit
    n_past_fwd = pli.n_past_fwd
    status = p7_PipelineFromFilterAndForwardScores(
        pli, om, bg, sq, NULL, th,
        usc, filtersc, vfsc, fwdsc, supplied_xmx, supplied_count,
    )
    if corruption == 5 and fesetround(original_rounding) != 0:
        free(original_xmx)
        free(forward_xmx)
        return eslEINVAL
    matrix_changed = memcmp(
        pli.oxf.xmx, original_xmx, xmx_count * sizeof(float),
    )
    ret_changed[0] = (
        pli.n_past_msv != n_past_msv
        or pli.n_past_bias != n_past_bias
        or pli.n_past_vit != n_past_vit
        or pli.n_past_fwd != n_past_fwd
        or th.N != 0
        or matrix_changed != 0
    )
    free(original_xmx)
    free(forward_xmx)
    return status


cdef int _invalid_simple_regions(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ* sq,
    P7_TOPHITS* th,
    int corruption,
    uint64_t* ret_changed,
) noexcept nogil:
    cdef P7_PIPELINE_SIMPLE_REGION regions[2]
    cdef P7_PIPELINE invalid_pli
    cdef P7_PIPELINE* supplied_pli = pli
    cdef p7_pipeline_domain_route_e route = p7_DOMAIN_SIMPLE
    cdef _double_bits f1_bits
    cdef _double_bits f2_bits
    cdef _double_bits f3_bits
    cdef _float_bits bad_float
    cdef const P7_PIPELINE_SIMPLE_REGION* supplied_regions = regions
    cdef unsigned char* snapshot = NULL
    cdef size_t offset = 0
    cdef size_t snapshot_size
    cdef uint64_t region_count = 1
    cdef float nexpected = 1.0
    cdef float filtersc
    cdef float fwdsc
    cdef float usc
    cdef float vfsc
    cdef int generation_bias_filter
    cdef int original_rounding = -1
    cdef int status

    ret_changed[0] = 1
    if sq.n < 4:
        return eslEINVAL
    status = p7_pli_NewModel(pli, om, bg)
    if status != eslOK:
        return status
    status = p7_pli_NewSeq(pli, sq)
    if status != eslOK:
        return status
    status = p7_bg_SetLength(bg, sq.n)
    if status != eslOK:
        return status
    status = p7_oprofile_ReconfigLength(om, sq.n)
    if status != eslOK:
        return status
    status = p7_omx_GrowTo(pli.oxf, om.M, 0, sq.n)
    if status != eslOK:
        return status
    status = p7_MSVFilter(sq.dsq, sq.n, om, pli.oxf, &usc)
    if status != eslOK:
        return status
    status = p7_bg_FilterScore(bg, sq.dsq, sq.n, &filtersc)
    if status != eslOK:
        return status
    status = p7_ViterbiFilter(sq.dsq, sq.n, om, pli.oxf, &vfsc)
    if status != eslOK:
        return status
    status = p7_ForwardParser(sq.dsq, sq.n, om, pli.oxf, &fwdsc)
    if status != eslOK:
        return status

    regions[0].i = 1
    regions[0].j = <uint32_t> sq.n
    regions[1].i = 3
    regions[1].j = <uint32_t> sq.n
    f1_bits.value = pli.F1
    f2_bits.value = pli.F2
    f3_bits.value = pli.F3
    generation_bias_filter = pli.do_biasfilter

    if corruption == 0:
        route = <p7_pipeline_domain_route_e> 0
    elif corruption == 1:
        region_count = 0
    elif corruption == 2:
        route = p7_DOMAIN_NO_REGIONS
    elif corruption == 3:
        regions[0].i = 0
    elif corruption == 4:
        regions[0].i = 2
        regions[0].j = 1
    elif corruption == 5:
        regions[0].j = <uint32_t> sq.n + 1
    elif corruption == 6:
        regions[0].j = 3
        regions[1].i = 3
        region_count = 2
    elif corruption == 7:
        regions[0].i = 3
        regions[0].j = 4
        regions[1].i = 1
        regions[1].j = 2
        region_count = 2
    elif corruption == 8:
        bad_float.bits = 0x7fc00000
        nexpected = bad_float.value
    elif corruption == 9:
        nexpected = -1.0
    elif corruption == 10:
        f1_bits.bits ^= 1
    elif corruption == 11:
        f2_bits.bits ^= 1
    elif corruption == 12:
        f3_bits.bits ^= 1
    elif corruption == 13:
        generation_bias_filter = 1 - pli.do_biasfilter
    elif corruption == 14:
        generation_bias_filter = 2
    elif corruption == 15:
        bad_float.bits = 0x7fc00000
        fwdsc = bad_float.value
    elif corruption == 16:
        pli.F2 = 0.0
        f2_bits.value = pli.F2
        bad_float.bits = 0xff800000
        vfsc = bad_float.value
    elif corruption == 17:
        bad_float.bits = 0x7fc00000
        filtersc = bad_float.value
    elif corruption == 18:
        pli.ddef.nregions = 1
    elif corruption == 19:
        om.L -= 1
    elif corruption == 20:
        original_rounding = fegetround()
        if fesetround(FE_DOWNWARD) != 0:
            return eslEINVAL
    elif corruption == 21:
        nexpected = <float> sq.n + 1.0
    elif corruption == 22:
        status = p7_oprofile_ReconfigUnihit(om, sq.n)
        if status != eslOK:
            return status
    elif corruption == 23:
        status = p7_oprofile_ReconfigLength(om, sq.n - 1)
        if status != eslOK:
            return status
        om.L = sq.n
    elif corruption == 24:
        status = p7_bg_SetLength(bg, sq.n - 1)
        if status != eslOK:
            return status
    elif corruption == 25:
        bad_float.value = bg.p1
        bad_float.bits ^= 1
        bg.p1 = bad_float.value
    elif corruption == 26:
        om.mode = 3
    elif corruption == 27:
        bad_float.bits = 0xff800000
        vfsc = bad_float.value
        route = p7_DOMAIN_NO_REGIONS
        nexpected = 0.0
        supplied_regions = NULL
        region_count = 0
    elif corruption == 28:
        bad_float.bits = 0x7fc00000
        pli.F1 = bad_float.value
        f1_bits.value = pli.F1
    elif corruption == 29:
        pli.F2 = 1.5
        f2_bits.value = pli.F2
    elif corruption == 30:
        pli.F3 = -1.0
        f3_bits.value = pli.F3
    elif corruption == 31:
        bad_float.bits = 0x7fc00000
        om.evparam[0] = bad_float.value
    elif corruption == 32:
        om.evparam[1] = 0.0
    elif corruption == 33:
        bad_float.bits = 0x7fc00000
        bg.omega = bad_float.value
    elif corruption == 34:
        bg.omega = 1.0
    elif corruption == 35:
        memcpy(&invalid_pli, pli, sizeof(P7_PIPELINE))
        invalid_pli.fwd = NULL
        supplied_pli = &invalid_pli
    elif corruption == 36:
        memcpy(&invalid_pli, pli, sizeof(P7_PIPELINE))
        invalid_pli.bck = NULL
        supplied_pli = &invalid_pli
    else:
        return eslEINVAL

    snapshot_size = (
        sizeof(P7_PIPELINE) + sizeof(P7_OPROFILE) + sizeof(P7_BG)
        + sizeof(P7_TOPHITS) + sizeof(P7_DOMAINDEF)
        + 4 * sizeof(P7_OMX)
    )
    snapshot = <unsigned char*> malloc(snapshot_size)
    if snapshot == NULL:
        if original_rounding != -1:
            fesetround(original_rounding)
        return eslEMEM
    memcpy(snapshot + offset, supplied_pli, sizeof(P7_PIPELINE))
    offset += sizeof(P7_PIPELINE)
    memcpy(snapshot + offset, om, sizeof(P7_OPROFILE))
    offset += sizeof(P7_OPROFILE)
    memcpy(snapshot + offset, bg, sizeof(P7_BG))
    offset += sizeof(P7_BG)
    memcpy(snapshot + offset, th, sizeof(P7_TOPHITS))
    offset += sizeof(P7_TOPHITS)
    memcpy(snapshot + offset, pli.ddef, sizeof(P7_DOMAINDEF))
    offset += sizeof(P7_DOMAINDEF)
    memcpy(snapshot + offset, pli.oxf, sizeof(P7_OMX))
    offset += sizeof(P7_OMX)
    memcpy(snapshot + offset, pli.oxb, sizeof(P7_OMX))
    offset += sizeof(P7_OMX)
    memcpy(snapshot + offset, pli.fwd, sizeof(P7_OMX))
    offset += sizeof(P7_OMX)
    memcpy(snapshot + offset, pli.bck, sizeof(P7_OMX))

    status = p7_PipelineFromFilterAndForwardSimpleRegions(
        supplied_pli, om, bg, sq, NULL, th,
        usc, filtersc, vfsc, fwdsc,
        f1_bits.bits, f2_bits.bits, f3_bits.bits,
        generation_bias_filter, route, nexpected,
        supplied_regions, region_count,
    )
    if original_rounding != -1 and fesetround(original_rounding) != 0:
        free(snapshot)
        return eslEINVAL

    offset = 0
    ret_changed[0] = memcmp(
        snapshot + offset, supplied_pli, sizeof(P7_PIPELINE),
    ) != 0
    offset += sizeof(P7_PIPELINE)
    ret_changed[0] |= memcmp(snapshot + offset, om, sizeof(P7_OPROFILE)) != 0
    offset += sizeof(P7_OPROFILE)
    ret_changed[0] |= memcmp(snapshot + offset, bg, sizeof(P7_BG)) != 0
    offset += sizeof(P7_BG)
    ret_changed[0] |= memcmp(snapshot + offset, th, sizeof(P7_TOPHITS)) != 0
    offset += sizeof(P7_TOPHITS)
    ret_changed[0] |= memcmp(
        snapshot + offset, pli.ddef, sizeof(P7_DOMAINDEF),
    ) != 0
    offset += sizeof(P7_DOMAINDEF)
    ret_changed[0] |= memcmp(snapshot + offset, pli.oxf, sizeof(P7_OMX)) != 0
    offset += sizeof(P7_OMX)
    ret_changed[0] |= memcmp(snapshot + offset, pli.oxb, sizeof(P7_OMX)) != 0
    offset += sizeof(P7_OMX)
    ret_changed[0] |= memcmp(snapshot + offset, pli.fwd, sizeof(P7_OMX)) != 0
    offset += sizeof(P7_OMX)
    ret_changed[0] |= memcmp(snapshot + offset, pli.bck, sizeof(P7_OMX)) != 0
    free(snapshot)
    return status


cdef void _restore_compact_alphabet_corruption(
    P7_OPROFILE* om,
    P7_BG* bg,
    ESL_SQ* sq,
    int corruption,
    int saved_int,
    char saved_char,
    ESL_DSQ saved_code,
    char* saved_sym,
    char** saved_degen,
    char* saved_degen_row,
    int* saved_ndegen,
    ESL_DSQ* saved_complement,
    const ESL_ALPHABET* saved_filter_alphabet,
    const ESL_ALPHABET* saved_sequence_alphabet,
) noexcept nogil:
    cdef ESL_ALPHABET* alphabet = <ESL_ALPHABET*> om.abc

    if corruption == 57:
        alphabet.type = saved_int
    elif corruption == 58:
        alphabet.K = saved_int
    elif corruption == 59:
        alphabet.Kp = saved_int
    elif corruption == 60:
        alphabet.sym[0] = saved_char
    elif corruption == 61:
        alphabet.inmap[65] = saved_code
    elif corruption == 62:
        alphabet.degen[0][0] = saved_char
    elif corruption == 63:
        alphabet.ndegen[0] = saved_int
    elif corruption == 64:
        alphabet.complement = saved_complement
    elif corruption == 65:
        bg.fhmm.K = saved_int
    elif corruption == 66:
        alphabet.sym = saved_sym
    elif corruption == 67:
        alphabet.degen = saved_degen
    elif corruption == 68:
        alphabet.degen[0] = saved_degen_row
    elif corruption == 69:
        alphabet.ndegen = saved_ndegen
    elif corruption == 70:
        bg.fhmm.abc = saved_filter_alphabet
    elif corruption == 71 or corruption == 72:
        sq.abc = <ESL_ALPHABET*> saved_sequence_alphabet


cdef int _invalid_compact_domains(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ* sq,
    P7_TOPHITS* th,
    int corruption,
    uint64_t* ret_changed,
) noexcept nogil:
    cdef P7_PIPELINE_SIMPLE_REGION region
    cdef P7_PIPELINE_COMPACT_DOMAIN domain
    cdef P7_PIPELINE_COMPACT_TRACE_STEP* traces = NULL
    cdef uint64_t trace_offsets[2]
    cdef float null2[29]
    cdef _float_bits bad_float
    cdef unsigned char* snapshot = NULL
    cdef size_t offset = 0
    cdef size_t n2sc_size
    cdef size_t dcl_size
    cdef size_t snapshot_size
    cdef uint64_t generation_tail_fingerprint
    cdef uint64_t final_target_count = 1
    cdef uint64_t trace_capacity
    cdef uint64_t trace_count
    cdef uint64_t trace_offset_count = 2
    cdef uint64_t supplied_trace_count
    cdef uint64_t supplied_null2_count = 29
    cdef float filtersc
    cdef float fwdsc
    cdef float nullsc
    cdef float sequence_correction
    cdef float sequence_compensation
    cdef float target_score
    cdef float domain_score
    cdef float reconstruction_score
    cdef float bias
    cdef float value
    cdef float y
    cdef float sum_next
    cdef float usc
    cdef float vfsc
    cdef double target_probability
    cdef double domain_probability
    cdef int first_match = -1
    cdef int original_rounding = -1
    cdef int status
    cdef int z
    cdef ESL_ALPHABET* alphabet
    cdef ESL_ALPHABET* replacement_alphabet = NULL
    cdef ESL_SQ* mutable_sq = <ESL_SQ*> sq
    cdef int saved_alphabet_int = 0
    cdef char saved_alphabet_char = 0
    cdef ESL_DSQ saved_alphabet_code = 0
    cdef char* saved_alphabet_sym = NULL
    cdef char** saved_alphabet_degen = NULL
    cdef char* saved_alphabet_degen_row = NULL
    cdef int* saved_alphabet_ndegen = NULL
    cdef ESL_DSQ* saved_alphabet_complement = NULL
    cdef const ESL_ALPHABET* saved_filter_alphabet = NULL
    cdef const ESL_ALPHABET* saved_sequence_alphabet = NULL

    ret_changed[0] = 1
    if sq.n <= 0:
        return eslEINVAL
    status = p7_pli_NewModel(pli, om, bg)
    if status != eslOK:
        return status
    status = p7_pli_NewSeq(pli, sq)
    if status != eslOK:
        return status
    status = p7_bg_SetLength(bg, sq.n)
    if status != eslOK:
        return status
    status = p7_oprofile_ReconfigLength(om, sq.n)
    if status != eslOK:
        return status
    status = p7_omx_GrowTo(pli.oxf, om.M, 0, sq.n)
    if status != eslOK:
        return status
    status = p7_MSVFilter(sq.dsq, sq.n, om, pli.oxf, &usc)
    if status != eslOK:
        return status
    status = p7_bg_FilterScore(bg, sq.dsq, sq.n, &filtersc)
    if status != eslOK:
        return status
    status = p7_ViterbiFilter(sq.dsq, sq.n, om, pli.oxf, &vfsc)
    if status != eslOK:
        return status
    status = p7_ForwardParser(sq.dsq, sq.n, om, pli.oxf, &fwdsc)
    if status != eslOK:
        return status

    region.i = 1
    region.j = <uint32_t> sq.n
    trace_capacity = <uint64_t> sq.n + <uint64_t> om.M + 6
    traces = <P7_PIPELINE_COMPACT_TRACE_STEP*> malloc(
        trace_capacity * sizeof(P7_PIPELINE_COMPACT_TRACE_STEP)
    )
    if traces == NULL:
        return eslEMEM
    status = _build_compact_domains(
        pli, om, sq, &region, 1, 0, 0, 0,
        &domain, null2, trace_offsets, traces,
        trace_capacity, &trace_count,
    )
    if status != eslOK:
        free(traces)
        return status
    supplied_trace_count = trace_count
    generation_tail_fingerprint = p7_pipeline_CompactTailFingerprintV2(pli)
    alphabet = <ESL_ALPHABET*> om.abc
    for z in range(<int> trace_count):
        if traces[z].state == p7T_M:
            first_match = z
            break
    if first_match < 0:
        free(traces)
        return eslENORESULT

    if corruption == 0:
        domain.status = 255
    elif corruption == 1:
        domain.action = 0
    elif corruption == 2:
        domain.has_own_scales = 1
    elif corruption == 3:
        domain.reserved = 1
    elif corruption == 4:
        domain.reserved2 = 1
    elif corruption == 5:
        domain.row_index = 1
    elif corruption == 6:
        domain.profile_index = 1
    elif corruption == 7:
        domain.sequence_index = 1
    elif corruption == 8:
        domain.envelope_begin = 0
    elif corruption == 9:
        domain.envelope_begin = domain.envelope_end + 1
    elif corruption == 10:
        domain.envelope_end = <uint32_t> sq.n + 1
    elif corruption == 11:
        domain.alignment_begin = 0
    elif corruption == 12:
        domain.model_begin = 0
    elif corruption == 13:
        bad_float.bits = 0x7fc00000
        domain.forward_score = bad_float.value
    elif corruption == 14:
        bad_float.bits = 0x7fc00000
        domain.backward_score = bad_float.value
    elif corruption == 15:
        bad_float.bits = 0x7fc00000
        domain.oa_score = bad_float.value
    elif corruption == 16:
        domain.oa_score = -1.0
    elif corruption == 17:
        bad_float.bits = 0x7fc00000
        domain.domain_correction = bad_float.value
    elif corruption == 18:
        domain.score_consistency = 0.003
    elif corruption == 19:
        domain.score_consistency += 0.0001
    elif corruption == 20:
        trace_offset_count = 1
    elif corruption == 21:
        trace_offsets[1] = trace_offsets[0]
    elif corruption == 22:
        supplied_trace_count -= 1
    elif corruption == 23:
        traces[0].state = <uint8_t> p7T_M
    elif corruption == 24:
        traces[first_match].sequence_position += 1
    elif corruption == 25:
        bad_float.bits = 0x7fc00000
        traces[first_match].posterior = bad_float.value
    elif corruption == 26:
        traces[first_match].reserved[0] = 1
    elif corruption == 27:
        null2[0] = 0.0
    elif corruption == 28:
        null2[21] += 0.01
    elif corruption == 29:
        generation_tail_fingerprint ^= 1
    elif corruption == 30:
        original_rounding = fegetround()
        if fesetround(FE_DOWNWARD) != 0:
            free(traces)
            return eslEINVAL
    elif corruption == 31:
        pli.E += 1.0
    elif corruption == 32:
        pli.ddef.nregions = 1
    elif corruption == 33:
        om.L -= 1
    elif corruption == 34:
        status = p7_oprofile_ReconfigUnihit(om, sq.n)
        if status != eslOK:
            free(traces)
            return status
    elif corruption == 35:
        status = p7_bg_SetLength(bg, sq.n - 1)
        if status != eslOK:
            free(traces)
            return status
    elif corruption == 36:
        domain.domain_correction += 0.01
    elif corruption == 37:
        domain.backward_score += 0.01
    elif corruption == 38:
        trace_offsets[0] = 1
    elif corruption == 39:
        supplied_null2_count -= 1
    elif corruption == 40:
        domain.oa_score += 0.01
    elif corruption == 42:
        final_target_count = 0
    elif corruption == 43:
        final_target_count = 9007199254740993
    elif corruption == 44:
        final_target_count = 2
        pli.Z = 1.5
        generation_tail_fingerprint = (
            p7_pipeline_CompactTailFingerprintV2(pli)
        )
    elif corruption == 45:
        pli.Z = 0.0
        generation_tail_fingerprint = (
            p7_pipeline_CompactTailFingerprintV2(pli)
        )
    elif corruption == 46:
        pli.Z = 2.0
        generation_tail_fingerprint = (
            p7_pipeline_CompactTailFingerprintV2(pli)
        )
    elif 47 <= corruption <= 56:
        status = p7_bg_NullOne(bg, sq.dsq, sq.n, &nullsc)
        if status != eslOK:
            free(traces)
            return status

        sequence_correction = 0.0
        sequence_compensation = 0.0
        for z in range(sq.n + 1):
            value = 0.0 if z == 0 else logf(null2[sq.dsq[z]])
            y = value - sequence_compensation
            sum_next = sequence_correction + y
            sequence_compensation = (
                (sum_next-sequence_correction)-y
            )
            sequence_correction = sum_next

        if pli.do_null2:
            bias = p7_FLogsum(
                0.0, log(bg.omega) + sequence_correction,
            )
        else:
            bias = 0.0
        target_score = (fwdsc - (nullsc + bias)) / eslCONST_LOG2

        if (
            (pli.do_null2 and
             domain.forward_score-domain.domain_correction > 0.0)
            or (not pli.do_null2 and domain.forward_score > 0.0)
        ):
            if pli.do_null2:
                bias = p7_FLogsum(
                    0.0, log(bg.omega) + domain.domain_correction,
                )
            else:
                bias = 0.0
            reconstruction_score = (
                domain.forward_score - (nullsc + bias)
            ) / eslCONST_LOG2
            if reconstruction_score > target_score:
                target_score = reconstruction_score

        if pli.do_null2:
            bias = p7_FLogsum(
                0.0, log(bg.omega) + domain.domain_correction,
            )
        else:
            bias = 0.0
        domain_score = (
            domain.forward_score - (nullsc + bias)
        ) / eslCONST_LOG2
        target_probability = exp(esl_exp_logsurv(
            target_score,
            om.evparam[<int> p7_FTAU],
            om.evparam[<int> p7_FLAMBDA],
        ))
        domain_probability = exp(esl_exp_logsurv(
            domain_score,
            om.evparam[<int> p7_FTAU],
            om.evparam[<int> p7_FLAMBDA],
        ))

        final_target_count = 7
        pli.use_bit_cutoffs = 0
        pli.by_E = False
        pli.inc_by_E = False
        pli.dom_by_E = False
        pli.incdom_by_E = False
        pli.T = -1.0e300
        pli.incT = -1.0e300
        pli.domT = -1.0e300
        pli.incdomT = -1.0e300
        if corruption == 47:
            pli.T = target_score
        elif corruption == 48:
            pli.incT = target_score
        elif corruption == 49:
            pli.domT = domain_score
        elif corruption == 50:
            pli.incdomT = domain_score
        elif corruption == 55:
            pli.T = target_score + 0.001
        else:
            pli.by_E = True
            pli.inc_by_E = True
            pli.dom_by_E = True
            pli.incdom_by_E = True
            pli.E = 1.0e300
            pli.incE = 1.0e300
            pli.domE = 1.0e300
            pli.incdomE = 1.0e300
            if corruption == 51:
                pli.Z_setby = p7_ZSETBY_NTARGETS
                pli.E = target_probability * 7.0
            elif corruption == 52:
                pli.Z_setby = p7_ZSETBY_OPTION
                pli.Z = 5.0
                pli.incE = target_probability * 5.0
            elif corruption == 53:
                pli.domZ_setby = p7_ZSETBY_NTARGETS
                pli.domE = domain_probability * 3.0
            elif corruption == 54:
                pli.domZ_setby = p7_ZSETBY_OPTION
                pli.domZ = 5.0
                pli.incdomE = domain_probability * 5.0
            else:
                pli.domZ_setby = p7_ZSETBY_NTARGETS
                pli.domE = domain_probability * 3.0 * (1.0 + 1.0e-6)
        generation_tail_fingerprint = (
            p7_pipeline_CompactTailFingerprintV2(pli)
        )
    elif corruption == 57:
        saved_alphabet_int = alphabet.type
        alphabet.type = eslDNA
    elif corruption == 58:
        saved_alphabet_int = alphabet.K
        alphabet.K = 19
    elif corruption == 59:
        saved_alphabet_int = alphabet.Kp
        alphabet.Kp = 28
    elif corruption == 60:
        saved_alphabet_char = alphabet.sym[0]
        alphabet.sym[0] = 90
    elif corruption == 61:
        saved_alphabet_code = alphabet.inmap[65]
        alphabet.inmap[65] += 1
    elif corruption == 62:
        saved_alphabet_char = alphabet.degen[0][0]
        alphabet.degen[0][0] = 0
    elif corruption == 63:
        saved_alphabet_int = alphabet.ndegen[0]
        alphabet.ndegen[0] = 2
    elif corruption == 64:
        saved_alphabet_complement = alphabet.complement
        alphabet.complement = <ESL_DSQ*> alphabet.sym
    elif corruption == 65:
        saved_alphabet_int = bg.fhmm.K
        bg.fhmm.K -= 1
    elif corruption == 66:
        saved_alphabet_sym = alphabet.sym
        alphabet.sym = NULL
    elif corruption == 67:
        saved_alphabet_degen = alphabet.degen
        alphabet.degen = NULL
    elif corruption == 68:
        saved_alphabet_degen_row = alphabet.degen[0]
        alphabet.degen[0] = NULL
    elif corruption == 69:
        saved_alphabet_ndegen = alphabet.ndegen
        alphabet.ndegen = NULL
    elif corruption == 70:
        saved_filter_alphabet = bg.fhmm.abc
        bg.fhmm.abc = NULL
    elif corruption == 71:
        saved_sequence_alphabet = <const ESL_ALPHABET*> mutable_sq.abc
        replacement_alphabet = esl_alphabet_Create(eslDNA)
        if replacement_alphabet == NULL:
            free(traces)
            return eslEMEM
        mutable_sq.abc = replacement_alphabet
    elif corruption == 72:
        saved_sequence_alphabet = <const ESL_ALPHABET*> mutable_sq.abc
        replacement_alphabet = esl_alphabet_CreateCustom(
            "ACDEFGHIKLMNPQRSTVWY-BJZOUX*~", 20, 29,
        )
        if replacement_alphabet == NULL:
            free(traces)
            return eslEMEM
        mutable_sq.abc = replacement_alphabet
    elif corruption != 41:
        free(traces)
        return eslEINVAL

    n2sc_size = (<size_t> pli.ddef.Lalloc + 1) * sizeof(float)
    dcl_size = <size_t> pli.ddef.nalloc * sizeof(P7_DOMAIN)
    snapshot_size = (
        sizeof(P7_PIPELINE) + sizeof(P7_OPROFILE) + sizeof(P7_BG)
        + sizeof(P7_TOPHITS) + sizeof(P7_DOMAINDEF)
        + sizeof(P7_TRACE) + 4 * sizeof(P7_OMX)
        + n2sc_size + dcl_size
    )
    snapshot = <unsigned char*> malloc(snapshot_size)
    if snapshot == NULL:
        _restore_compact_alphabet_corruption(
            om, bg, mutable_sq, corruption,
            saved_alphabet_int, saved_alphabet_char, saved_alphabet_code,
            saved_alphabet_sym, saved_alphabet_degen,
            saved_alphabet_degen_row, saved_alphabet_ndegen,
            saved_alphabet_complement, saved_filter_alphabet,
            saved_sequence_alphabet,
        )
        esl_alphabet_Destroy(replacement_alphabet)
        if original_rounding != -1:
            fesetround(original_rounding)
        free(traces)
        return eslEMEM
    memcpy(snapshot + offset, pli, sizeof(P7_PIPELINE))
    offset += sizeof(P7_PIPELINE)
    memcpy(snapshot + offset, om, sizeof(P7_OPROFILE))
    offset += sizeof(P7_OPROFILE)
    memcpy(snapshot + offset, bg, sizeof(P7_BG))
    offset += sizeof(P7_BG)
    memcpy(snapshot + offset, th, sizeof(P7_TOPHITS))
    offset += sizeof(P7_TOPHITS)
    memcpy(snapshot + offset, pli.ddef, sizeof(P7_DOMAINDEF))
    offset += sizeof(P7_DOMAINDEF)
    memcpy(snapshot + offset, pli.ddef.tr, sizeof(P7_TRACE))
    offset += sizeof(P7_TRACE)
    memcpy(snapshot + offset, pli.oxf, sizeof(P7_OMX))
    offset += sizeof(P7_OMX)
    memcpy(snapshot + offset, pli.oxb, sizeof(P7_OMX))
    offset += sizeof(P7_OMX)
    memcpy(snapshot + offset, pli.fwd, sizeof(P7_OMX))
    offset += sizeof(P7_OMX)
    memcpy(snapshot + offset, pli.bck, sizeof(P7_OMX))
    offset += sizeof(P7_OMX)
    memcpy(snapshot + offset, pli.ddef.n2sc, n2sc_size)
    offset += n2sc_size
    memcpy(snapshot + offset, pli.ddef.dcl, dcl_size)

    status = p7_PipelineFromFilterForwardAndCompactDomainsV2(
        pli, om, bg, sq, NULL, th,
        usc, filtersc, vfsc, fwdsc,
        generation_tail_fingerprint,
        final_target_count,
        0, 0, 0, 1.0,
        &domain, 1,
        trace_offsets, trace_offset_count,
        traces, supplied_trace_count,
        null2, supplied_null2_count,
    )
    _restore_compact_alphabet_corruption(
        om, bg, mutable_sq, corruption,
        saved_alphabet_int, saved_alphabet_char, saved_alphabet_code,
        saved_alphabet_sym, saved_alphabet_degen,
        saved_alphabet_degen_row, saved_alphabet_ndegen,
        saved_alphabet_complement, saved_filter_alphabet,
        saved_sequence_alphabet,
    )
    esl_alphabet_Destroy(replacement_alphabet)
    if original_rounding != -1 and fesetround(original_rounding) != 0:
        free(snapshot)
        free(traces)
        return eslEINVAL

    offset = 0
    ret_changed[0] = memcmp(snapshot + offset, pli, sizeof(P7_PIPELINE)) != 0
    offset += sizeof(P7_PIPELINE)
    ret_changed[0] |= memcmp(snapshot + offset, om, sizeof(P7_OPROFILE)) != 0
    offset += sizeof(P7_OPROFILE)
    ret_changed[0] |= memcmp(snapshot + offset, bg, sizeof(P7_BG)) != 0
    offset += sizeof(P7_BG)
    ret_changed[0] |= memcmp(snapshot + offset, th, sizeof(P7_TOPHITS)) != 0
    offset += sizeof(P7_TOPHITS)
    ret_changed[0] |= memcmp(
        snapshot + offset, pli.ddef, sizeof(P7_DOMAINDEF),
    ) != 0
    offset += sizeof(P7_DOMAINDEF)
    ret_changed[0] |= memcmp(
        snapshot + offset, pli.ddef.tr, sizeof(P7_TRACE),
    ) != 0
    offset += sizeof(P7_TRACE)
    ret_changed[0] |= memcmp(snapshot + offset, pli.oxf, sizeof(P7_OMX)) != 0
    offset += sizeof(P7_OMX)
    ret_changed[0] |= memcmp(snapshot + offset, pli.oxb, sizeof(P7_OMX)) != 0
    offset += sizeof(P7_OMX)
    ret_changed[0] |= memcmp(snapshot + offset, pli.fwd, sizeof(P7_OMX)) != 0
    offset += sizeof(P7_OMX)
    ret_changed[0] |= memcmp(snapshot + offset, pli.bck, sizeof(P7_OMX)) != 0
    offset += sizeof(P7_OMX)
    if pli.ddef.n2sc != NULL:
        ret_changed[0] |= memcmp(
            snapshot + offset, pli.ddef.n2sc, n2sc_size,
        ) != 0
    offset += n2sc_size
    if pli.ddef.dcl != NULL:
        ret_changed[0] |= memcmp(
            snapshot + offset, pli.ddef.dcl, dcl_size,
        ) != 0
    free(snapshot)
    free(traces)
    return status


def search_from_msv(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
):
    cdef TopHits hits = TopHits(query)
    cdef int status

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        status = _loop_msv(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ**> sequences._refs,
            sequences._length,
            hits._th,
        )
        if status == eslOK:
            hits._sort_by_key()
            hits._threshold(pipeline)
    if status != eslOK:
        raise RuntimeError(f"HMMER status {status}")
    hits._query = query
    hits._empty = False
    return hits


def search(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    int gpu_viterbi_status=0,
    bint poison_filtersc=False,
):
    cdef TopHits hits = TopHits(query)
    cdef int status

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        status = _loop(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ**> sequences._refs,
            sequences._length,
            hits._th,
            gpu_viterbi_status,
            poison_filtersc,
        )
        if status == eslOK:
            hits._sort_by_key()
            hits._threshold(pipeline)
    if status != eslOK:
        raise RuntimeError(f"HMMER status {status}")
    hits._query = query
    hits._empty = False
    return hits


def search_forward(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    bint omit_forward_matrix=False,
):
    cdef TopHits hits = TopHits(query)
    cdef int status

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        status = _loop_forward(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ**> sequences._refs,
            sequences._length,
            hits._th,
            omit_forward_matrix,
        )
        if status == eslOK:
            hits._sort_by_key()
            hits._threshold(pipeline)
    if status != eslOK:
        raise RuntimeError(f"HMMER status {status}")
    hits._query = query
    hits._empty = False
    return hits


def search_simple_regions(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
):
    """Exercise guarded simple-region continuation with a CPU region oracle."""
    cdef TopHits hits = TopHits(query)
    cdef uint64_t route_counts[3]
    cdef int status

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        status = _loop_simple_regions(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ**> sequences._refs,
            sequences._length,
            hits._th,
            route_counts,
        )
        if status == eslOK:
            hits._sort_by_key()
            hits._threshold(pipeline)
    if status != eslOK:
        raise RuntimeError(f"HMMER status {status}")
    hits._query = query
    hits._empty = False
    return hits, (
        route_counts[0], route_counts[1], route_counts[2],
    )


cdef bint _compact_alphabet_contract_case(int fault) except -1:
    cdef ESL_ALPHABET* profile = esl_alphabet_Create(eslAMINO)
    cdef ESL_ALPHABET* background = esl_alphabet_Create(eslAMINO)
    cdef ESL_ALPHABET* filter_abc = esl_alphabet_Create(eslAMINO)
    cdef ESL_ALPHABET* sequence = esl_alphabet_Create(eslAMINO)
    cdef char* saved_degen = NULL
    cdef char** saved_degen_table = NULL
    cdef int* saved_ndegen = NULL
    cdef bint result

    if (
        profile == NULL
        or background == NULL
        or filter_abc == NULL
        or sequence == NULL
    ):
        esl_alphabet_Destroy(profile)
        esl_alphabet_Destroy(background)
        esl_alphabet_Destroy(filter_abc)
        esl_alphabet_Destroy(sequence)
        raise MemoryError("alphabet contract probe allocation failed")
    try:
        if fault == 1:
            profile.type = eslDNA
        elif fault == 2:
            profile.K = 19
        elif fault == 3:
            profile.Kp = 28
        elif fault == 4:
            profile.sym[0] = 90
        elif fault == 5:
            profile.sym[29] = 88
        elif fault == 6:
            background.inmap[65] += 1
        elif fault == 7:
            filter_abc.degen[0][0] = 0
        elif fault == 8:
            sequence.ndegen[0] = 2
        elif fault == 9:
            profile.complement = <unsigned char*> profile.sym
        elif fault == 10:
            saved_degen = profile.degen[0]
            profile.degen[0] = NULL
        elif fault == 11:
            saved_ndegen = profile.ndegen
            profile.ndegen = NULL
        elif fault == 12:
            esl_alphabet_Destroy(sequence)
            sequence = esl_alphabet_Create(eslDNA)
            if sequence == NULL:
                raise MemoryError("DNA alphabet contract probe allocation failed")
        elif fault == 13:
            saved_degen_table = profile.degen
            profile.degen = NULL
        elif fault == 14:
            saved_degen = profile.sym
            profile.sym = NULL
        elif fault == 15:
            esl_alphabet_Destroy(sequence)
            sequence = esl_alphabet_CreateCustom(
                "ACDEFGHIKLMNPQRSTVWY-BJZOUX*~", 20, 29,
            )
            if sequence == NULL:
                raise MemoryError(
                    "custom alphabet contract probe allocation failed"
                )
        elif fault != 0:
            raise ValueError("unknown alphabet contract fault")
        result = p7_compact_AlphabetSetIsExactV2(
            profile, background, filter_abc, sequence,
        )
        if fault == 0:
            result = (
                result
                and p7_compact_AlphabetIsCanonicalAminoV2(profile)
                and p7_compact_AlphabetsAreEqualV2(profile, background)
            )
        return result
    finally:
        if fault == 1:
            profile.type = eslAMINO
        elif fault == 2:
            profile.K = 20
        elif fault == 3:
            profile.Kp = 29
        elif fault == 4:
            profile.sym[0] = 65
        elif fault == 5:
            profile.sym[29] = 0
        elif fault == 6:
            background.inmap[65] -= 1
        elif fault == 7:
            filter_abc.degen[0][0] = 1
        elif fault == 8:
            sequence.ndegen[0] = 1
        elif fault == 9:
            profile.complement = NULL
        elif fault == 10:
            profile.degen[0] = saved_degen
        elif fault == 11:
            profile.ndegen = saved_ndegen
        elif fault == 13:
            profile.degen = saved_degen_table
        elif fault == 14:
            profile.sym = saved_degen
        esl_alphabet_Destroy(profile)
        esl_alphabet_Destroy(background)
        esl_alphabet_Destroy(filter_abc)
        esl_alphabet_Destroy(sequence)


cdef bint _compact_shared_alphabet_corruption_case() except -1:
    cdef ESL_ALPHABET* alphabet = esl_alphabet_Create(eslAMINO)
    cdef bint result
    if alphabet == NULL:
        raise MemoryError("shared alphabet contract probe allocation failed")
    try:
        alphabet.inmap[65] += 1
        result = p7_compact_AlphabetSetIsExactV2(
            alphabet, alphabet, alphabet, alphabet,
        )
        return result
    finally:
        alphabet.inmap[65] -= 1
        esl_alphabet_Destroy(alphabet)


def compact_alphabet_contract_cases():
    """Probe distinct-equal acceptance and fieldwise fail-closed behavior."""
    names = (
        "distinct_equal",
        "type",
        "K",
        "Kp",
        "symbols",
        "symbol_terminator",
        "input_map",
        "degeneracy_table",
        "degeneracy_count",
        "complement",
        "null_degeneracy_row",
        "null_degeneracy_counts",
        "dna_sequence",
        "null_degeneracy_table",
        "null_symbols",
        "custom_sequence",
    )
    result = {
        name: bool(_compact_alphabet_contract_case(fault))
        for fault, name in enumerate(names)
    }
    result["shared_corruption"] = bool(
        _compact_shared_alphabet_corruption_case()
    )
    return result


def search_compact_domains(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    float compact_fwd_delta=0.0,
):
    """Exercise compact-domain continuation with a pristine CPU oracle."""
    cdef TopHits hits = TopHits(query)
    cdef uint64_t route_counts[5]
    cdef int status

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        status = _loop_compact_domains(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ**> sequences._refs,
            sequences._length,
            hits._th,
            route_counts,
            compact_fwd_delta,
        )
        if status == eslOK:
            hits._sort_by_key()
            hits._threshold(pipeline)
    if status != eslOK:
        raise RuntimeError(f"HMMER status {status}")
    hits._query = query
    hits._empty = False
    return hits, (
        route_counts[0], route_counts[1],
        route_counts[2], route_counts[3], route_counts[4],
    )


def invalid_forward_status(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    int corruption,
):
    cdef TopHits hits
    cdef uint64_t changed
    cdef int status

    if sequences._length != 1:
        raise ValueError("invalid-forward probe requires one target")
    hits = TopHits(query)
    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        status = _invalid_forward(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ*> sequences._refs[0],
            hits._th,
            corruption,
            &changed,
        )
    return status, bool(changed)


def invalid_simple_regions_status(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    int corruption,
):
    cdef TopHits hits
    cdef uint64_t changed
    cdef int status

    if sequences._length != 1:
        raise ValueError("invalid simple-region probe requires one target")
    hits = TopHits(query)
    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        status = _invalid_simple_regions(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ*> sequences._refs[0],
            hits._th,
            corruption,
            &changed,
        )
    return status, bool(changed)


def invalid_compact_domains_status(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    int corruption,
):
    cdef TopHits hits
    cdef uint64_t changed
    cdef int status

    if sequences._length != 1:
        raise ValueError("invalid compact-domain probe requires one target")
    hits = TopHits(query)
    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        status = _invalid_compact_domains(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ*> sequences._refs[0],
            hits._th,
            corruption,
            &changed,
        )
    return status, bool(changed)


def forward_rescale_count(
    Pipeline pipeline,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
):
    cdef const ESL_SQ* sequence
    cdef float fwdsc
    cdef int rescales = 0
    cdef int status
    cdef int i
    cdef size_t t

    if sequences._length == 0:
        raise ValueError("Forward rescale probe requires a target")
    with nogil:
        status = eslOK
        for t in range(sequences._length):
            sequence = <const ESL_SQ*> sequences._refs[t]
            status = p7_oprofile_ReconfigLength(
                optimized_profile._om, sequence.n,
            )
            if status != eslOK:
                break
            status = p7_omx_GrowTo(
                pipeline._pli.oxf,
                optimized_profile._om.M,
                0,
                sequence.n,
            )
            if status != eslOK:
                break
            status = p7_ForwardParser(
                sequence.dsq,
                sequence.n,
                optimized_profile._om,
                pipeline._pli.oxf,
                &fwdsc,
            )
            if status != eslOK:
                break
            for i in range(1, sequence.n + 1):
                rescales += (
                    pipeline._pli.oxf.xmx[
                        i * FORWARD_SPECIAL_CELLS + 5
                    ]
                    > 1.0
                )
    if status != eslOK:
        raise RuntimeError(f"HMMER status {status}")
    return rescales
