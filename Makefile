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
CYTHON_CPP := $(BUILD_DIR)/cuda/_native.cpp
CYTHON_OBJ := $(BUILD_DIR)/cuda/_native.o
CUDA_OBJ := $(BUILD_DIR)/cuda/ssv_cuda.o
CUDA_MODULE := python/plan7_gpu/_native$(PYTHON_EXT_SUFFIX)
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

.PHONY: all oracle cuda cuda-test test clean

all: oracle

oracle: $(ORACLE_BIN)

cuda: $(CUDA_MODULE)

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

$(CUDA_OBJ): cuda/ssv_cuda.cu cuda/ssv_cuda.h
	mkdir -p $(@D)
	$(NVCC) -O3 -g -std=c++17 -Xcompiler=-fPIC $(CUDA_ARCH_FLAGS) \
		-Icuda -c -o $@ $<

$(CUDA_MODULE): $(CYTHON_OBJ) $(CUDA_OBJ)
	$(NVCC) -shared $(CUDA_ARCH_FLAGS) -o $@ $(CYTHON_OBJ) $(CUDA_OBJ)

test: oracle
	python3 -m unittest discover -s tests -v
	$(ORACLE_BIN) --max-models 1 --max-seqs 45 --strict \
		$(HMMER_ROOT)/tutorial/globins4.hmm \
		$(HMMER_ROOT)/tutorial/globins45.fa >/dev/null

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(CUDA_MODULE)
