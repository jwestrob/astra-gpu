HMMER_ROOT ?= refs/src/hmmer-3.4
BUILD_DIR  ?= build
CC         ?= gcc

CPPFLAGS += -I$(HMMER_ROOT)/src -I$(HMMER_ROOT)/easel -I$(HMMER_ROOT)/libdivsufsort
CFLAGS   += -O2 -g -std=c11 -Wall -Wextra -Wshadow -Wconversion
LDLIBS   += -L$(HMMER_ROOT)/src -lhmmer \
	-L$(HMMER_ROOT)/easel -leasel \
	-L$(HMMER_ROOT)/libdivsufsort -ldivsufsort -lpthread -lm

ORACLE_BIN := $(BUILD_DIR)/oracle/msv-oracle
ORACLE_OBJ := $(BUILD_DIR)/oracle/msv_oracle.o
HMMER_LIBS := $(HMMER_ROOT)/src/libhmmer.a \
	$(HMMER_ROOT)/easel/libeasel.a \
	$(HMMER_ROOT)/libdivsufsort/libdivsufsort.a
HMMER_HEADERS := $(HMMER_ROOT)/src/hmmer.h \
	$(HMMER_ROOT)/src/p7_config.h \
	$(HMMER_ROOT)/src/impl_sse/impl_sse.h

.PHONY: all oracle test clean

all: oracle

oracle: $(ORACLE_BIN)

$(ORACLE_BIN): $(ORACLE_OBJ) $(HMMER_LIBS)
	$(CC) $(LDFLAGS) -o $@ $(ORACLE_OBJ) $(LDLIBS)

$(ORACLE_OBJ): oracle/msv_oracle.c $(HMMER_HEADERS)
	mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

test: oracle
	python3 -m unittest discover -s tests -v
	$(ORACLE_BIN) --max-models 1 --max-seqs 45 --strict \
		$(HMMER_ROOT)/tutorial/globins4.hmm \
		$(HMMER_ROOT)/tutorial/globins45.fa >/dev/null

clean:
	rm -rf $(BUILD_DIR)
