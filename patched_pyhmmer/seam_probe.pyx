# cython: language_level=3

from libc.stddef cimport size_t

from libeasel cimport eslEINVAL, eslENORESULT, eslERANGE, eslOK
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
    p7_pipeline_Reuse,
    p7_pipeline_vitmode_e,
    p7_pli_NewModel,
    p7_pli_NewSeq,
)
from libhmmer.p7_tophits cimport P7_TOPHITS

from pyhmmer.easel cimport DigitalSequenceBlock
from pyhmmer.plan7 cimport HMM, OptimizedProfile, Pipeline, TopHits


cdef enum:
    GPU_VITERBI_NOT_RUN_C = -1

GPU_VITERBI_NOT_RUN = GPU_VITERBI_NOT_RUN_C
GPU_VITERBI_OK = eslOK
GPU_VITERBI_ERANGE = eslERANGE
GPU_VITERBI_ENORESULT = eslENORESULT


cdef extern from "impl_sse/impl_sse.h" nogil:
    int p7_ViterbiFilter(
        const unsigned char *dsq,
        int L,
        const P7_OPROFILE *om,
        P7_OMX *ox,
        float *ret_sc,
    )


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
