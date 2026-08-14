HMMER_ROOT ?= refs/src/hmmer-3.4
BUILD_DIR  ?= build
CC         ?= gcc
CXX        ?= g++
NVCC       ?= nvcc
PYTHON     ?= python3

CPPFLAGS += -I$(HMMER_ROOT)/src -I$(HMMER_ROOT)/easel -I$(HMMER_ROOT)/libdivsufsort
CFLAGS   += -O2 -g -std=c11 -Wall -Wextra -Wshadow -Wconversion
LDLIBS   += -L$(HMMER_ROOT)/src -lhmmer \
	-L$(HMMER_ROOT)/easel -leasel \
	-L$(HMMER_ROOT)/libdivsufsort -ldivsufsort -lpthread -lm

ORACLE_BIN := $(BUILD_DIR)/oracle/msv-oracle
ORACLE_OBJ := $(BUILD_DIR)/oracle/msv_oracle.o
PYTHON_EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix)
PYHMMER_LIB_DIR ?= $(shell $(PYTHON) -c 'from pathlib import Path; import pyhmmer; print((Path(pyhmmer.__file__).resolve().parent.parent / "pyhmmer.libs").resolve())')
PYHMMER_PACKAGE_DIR ?= $(shell $(PYTHON) -c 'from pathlib import Path; import pyhmmer; print(Path(pyhmmer.__file__).resolve().parent)')
PYHMMER_ABI_VERSION := 0.12.0
PYHMMER_HMMER_IMPL := SSE
PYHMMER_TARGET_SYSTEM := Linux
PYHMMER_SIMD_CFLAGS := -msse4.1
PYHMMER_CYTHON_INCLUDE ?= $(PYHMMER_LIB_DIR)/cython/include
PYHMMER_INCLUDE ?= $(PYHMMER_LIB_DIR)/include
PYHMMER_EASEL_INCLUDE ?= $(PYHMMER_LIB_DIR)/include/libeasel
PYHMMER_EASEL_LIB ?= $(PYHMMER_LIB_DIR)/liblibeasel.so
PYHMMER_HMMER_LIB ?= $(PYHMMER_LIB_DIR)/liblibhmmer.so
CYTHON_CPP := $(BUILD_DIR)/cuda/_native.cpp
CYTHON_OBJ := $(BUILD_DIR)/cuda/_native.o
CUDA_OBJ := $(BUILD_DIR)/cuda/ssv_cuda.o
CUDA_MODULE := python/plan7_gpu/_native$(PYTHON_EXT_SUFFIX)
PIPELINE_C := $(BUILD_DIR)/pipeline/_pipeline.c
PIPELINE_OBJ := $(BUILD_DIR)/pipeline/_pipeline.o
PIPELINE_MODULE := python/plan7_gpu/_pipeline$(PYTHON_EXT_SUFFIX)
CUDA_ARCH_FLAGS := \
	--generate-code=arch=compute_75,code=sm_75 \
	--generate-code=arch=compute_75,code=compute_75 \
	--generate-code=arch=compute_90,code=sm_90
HMMER_LIBS := $(HMMER_ROOT)/src/libhmmer.a \
	$(HMMER_ROOT)/easel/libeasel.a \
	$(HMMER_ROOT)/libdivsufsort/libdivsufsort.a
HMMER_HEADERS := $(HMMER_ROOT)/src/hmmer.h \
	$(HMMER_ROOT)/src/p7_config.h \
	$(HMMER_ROOT)/src/impl_sse/impl_sse.h

.PHONY: all oracle cuda cuda-test pipeline pipeline-test test clean

all: oracle

oracle: $(ORACLE_BIN)

cuda: $(CUDA_MODULE) $(PIPELINE_MODULE)

pipeline: $(PIPELINE_MODULE)

pipeline-test: pipeline
	PYTHONPATH=python $(PYTHON) -m unittest discover -s tests \
		-p 'test_masked_pipeline.py' -v

cuda-test: cuda
	PYTHONPATH=python $(PYTHON) -c 'import sys; from plan7_gpu import _native; sys.exit("no CUDA device") if _native.device_count() <= 0 else None'
	PYTHONPATH=python $(PYTHON) -m unittest discover -s tests -p 'test_cuda_ssv.py' -v

$(ORACLE_BIN): $(ORACLE_OBJ) $(HMMER_LIBS)
	$(CC) $(LDFLAGS) -o $@ $(ORACLE_OBJ) $(LDLIBS)

$(ORACLE_OBJ): oracle/msv_oracle.c $(HMMER_HEADERS)
	mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

$(CYTHON_CPP): python/plan7_gpu/_native.pyx cuda/ssv_cuda.h
	mkdir -p $(@D)
	$(PYTHON) -m cython --cplus -3 -o $@ $<

$(CYTHON_OBJ): $(CYTHON_CPP) cuda/ssv_cuda.h
	$(CXX) -O3 -g -std=c++17 -fPIC -Wall -Wextra -Icuda \
		$$($(PYTHON)-config --includes) -c -o $@ $<

$(CUDA_OBJ): cuda/ssv_cuda.cu cuda/ssv_cuda.h \
	$(PYHMMER_EASEL_INCLUDE)/easel.h \
	$(PYHMMER_EASEL_INCLUDE)/esl_gumbel.h
	mkdir -p $(@D)
	$(NVCC) -O3 -g -std=c++17 -Xcompiler=-fPIC $(CUDA_ARCH_FLAGS) \
		-Icuda -I$(PYHMMER_EASEL_INCLUDE) -c -o $@ $<

$(CUDA_MODULE): $(CYTHON_OBJ) $(CUDA_OBJ) $(PYHMMER_EASEL_LIB)
	$(NVCC) -shared $(CUDA_ARCH_FLAGS) -o $@ $(CYTHON_OBJ) $(CUDA_OBJ) \
		-L$(PYHMMER_LIB_DIR) -llibeasel \
		-Xlinker -rpath -Xlinker $(PYHMMER_LIB_DIR)

$(PIPELINE_C): python/plan7_gpu/_pipeline.pyx \
		$(PYHMMER_PACKAGE_DIR)/plan7.pxd \
		$(PYHMMER_PACKAGE_DIR)/easel.pxd \
		$(PYHMMER_CYTHON_INCLUDE)/libeasel/sq.pxd \
		$(PYHMMER_CYTHON_INCLUDE)/libhmmer/p7_bg.pxd \
		$(PYHMMER_CYTHON_INCLUDE)/libhmmer/p7_pipeline.pxd \
		$(PYHMMER_CYTHON_INCLUDE)/libhmmer/p7_tophits.pxd \
		$(PYHMMER_CYTHON_INCLUDE)/libhmmer/impl/p7_oprofile.pxd
	mkdir -p $(@D)
	test "$$($(PYTHON) -c 'import pyhmmer; print(pyhmmer.__version__)')" = "$(PYHMMER_ABI_VERSION)" || \
		{ echo "plan7_gpu._pipeline requires PyHMMER $(PYHMMER_ABI_VERSION)" >&2; exit 1; }
	$(PYTHON) -m cython -3 -I$(PYHMMER_CYTHON_INCLUDE) \
		-E HMMER_IMPL=$(PYHMMER_HMMER_IMPL) \
		-E TARGET_SYSTEM=$(PYHMMER_TARGET_SYSTEM) -o $@ $<

$(PIPELINE_OBJ): $(PIPELINE_C)
	$(CC) -O3 -g -std=c11 -fPIC $(PYHMMER_SIMD_CFLAGS) \
		$$($(PYTHON)-config --includes) \
		-I$(PYHMMER_INCLUDE) -I$(PYHMMER_EASEL_INCLUDE) \
		-I$(PYHMMER_INCLUDE)/libhmmer -c -o $@ $<

$(PIPELINE_MODULE): $(PIPELINE_OBJ) $(PYHMMER_HMMER_LIB) $(PYHMMER_EASEL_LIB)
	$(CC) -shared -o $@ $(PIPELINE_OBJ) -L$(PYHMMER_LIB_DIR) \
		-Wl,--no-as-needed -llibhmmer -llibeasel \
		-Wl,-rpath,$(PYHMMER_LIB_DIR)

test: oracle pipeline
	PYTHONPATH=python $(PYTHON) -m unittest discover -s tests -v
	$(ORACLE_BIN) --max-models 1 --max-seqs 45 --strict \
		$(HMMER_ROOT)/tutorial/globins4.hmm \
		$(HMMER_ROOT)/tutorial/globins45.fa >/dev/null

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(CUDA_MODULE)
	rm -f $(PIPELINE_MODULE)
