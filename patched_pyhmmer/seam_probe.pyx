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
)
from libhmmer.p7_bg cimport (
    P7_BG,
    p7_bg_FilterScore,
    p7_bg_SetLength,
)
from libhmmer.p7_pipeline cimport (
    P7_PIPELINE,
    p7_SEARCH_SEQS,
    p7_VIT_CPU,
    p7_VIT_EXTERNAL,
    p7_VIT_NONE,
    p7_PipelineFromMSV,
    p7_PipelineFromFilterScores,
    p7_PipelineFromFilterAndForwardScores,
    p7_pipeline_Reuse,
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


cdef union _float_bits:
    float value
    uint32_t bits


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
