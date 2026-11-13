override CFLAGS += -O3 -fPIC -Wall -Wextra

# Target shared library name based on OS
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    SHARED_EXT := dylib
    LDFLAGS := -dynamiclib
else
    SHARED_EXT := so
    LDFLAGS := -shared
endif

ifeq ($(UNAME_M),arm64)
    ARCH_FLAGS ?= -march=armv8-a+simd
else ifeq ($(UNAME_M),aarch64)
    ARCH_FLAGS ?= -march=armv8-a+simd
else ifeq ($(UNAME_M),x86_64)
    HAS_AVX2 := $(shell if [ "$$(uname -s)" = "Darwin" ]; then sysctl -n hw.optional.avx2_0 2>/dev/null || echo 0; else grep -c "avx2" /proc/cpuinfo 2>/dev/null || echo 0; fi)
    ifneq ($(HAS_AVX2),0)
        ARCH_FLAGS ?= -mavx2 -mfma
    else
        ARCH_FLAGS ?=
    endif
else
    ARCH_FLAGS ?=
endif

ifeq ($(UNAME_M),x86_64)
    AVX512_FLAGS ?= -mavx512f -mavx512dq -mavx512vl -mavx512bw
else
    AVX512_FLAGS ?=
endif

%_avx512.o: CFLAGS_ISA := $(AVX512_FLAGS)
%avx512.o: CFLAGS_ISA := $(AVX512_FLAGS)

TARGET := libparallax_c.$(SHARED_EXT)
SRCS := $(wildcard *.c)
OBJS := $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) $(ARCH_FLAGS) -o $@ $(OBJS) -lm

%.o: %.c
	$(CC) $(CFLAGS) $(ARCH_FLAGS) $(CFLAGS_ISA) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *.so *.dylib

.PHONY: all clean
