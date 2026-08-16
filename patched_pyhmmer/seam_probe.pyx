# cython: language_level=3

from libc.stddef cimport size_t
from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcmp, memcpy

from libeasel cimport eslEINVAL, eslEMEM, eslENORESULT, eslERANGE, eslOK
from libeasel.sq cimport ESL_SQ
from libhmmer.impl_sse cimport p7_MSVFilter
from libhmmer.impl.p7_omx cimport P7_OMX, p7_omx_GrowTo
from libhmmer.impl.p7_oprofile cimport (
    P7_OPROFILE,
    p7_oprofile_ReconfigLength,
    p7_oprofile_ReconfigUnihit,
)
from libhmmer.p7_domaindef cimport P7_DOMAINDEF, p7_domaindef_GrowTo
from libhmmer.p7_bg cimport (
    P7_BG,
    p7_bg_FilterScore,
    p7_bg_SetLength,
)
from libhmmer.p7_pipeline cimport (
    P7_PIPELINE,
    P7_PIPELINE_SIMPLE_REGION,
    p7_DOMAIN_NO_REGIONS,
    p7_DOMAIN_SIMPLE,
    p7_SEARCH_SEQS,
    p7_VIT_CPU,
    p7_VIT_EXTERNAL,
    p7_VIT_NONE,
    p7_PipelineFromMSV,
    p7_PipelineFromFilterScores,
    p7_PipelineFromFilterAndForwardScores,
    p7_PipelineFromFilterAndForwardSimpleRegions,
    p7_pipeline_Reuse,
    p7_pipeline_domain_route_e,
    p7_pipeline_vitmode_e,
    p7_pli_NewModel,
    p7_pli_NewSeq,
)
from libhmmer.p7_tophits cimport P7_TOPHITS

from pyhmmer.easel cimport DigitalSequenceBlock
from pyhmmer.plan7 cimport HMM, OptimizedProfile, Pipeline, TopHits


cdef extern from "fenv.h" nogil:
    int FE_DOWNWARD
    int fegetround()
    int fesetround(int round)


cdef enum:
    GPU_VITERBI_NOT_RUN_C = -1
    FORWARD_SPECIAL_CELLS = 6

GPU_VITERBI_NOT_RUN = GPU_VITERBI_NOT_RUN_C
GPU_VITERBI_OK = eslOK
GPU_VITERBI_ERANGE = eslERANGE
GPU_VITERBI_ENORESULT = eslENORESULT
HMMER_EINVAL = eslEINVAL


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
