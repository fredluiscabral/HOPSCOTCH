# Makefile — compila: serial, omp_naive, omp_semaphores (com barreira),
# omp_sem_nobarrier (sem barreira), busywait_barrier, busywait_nobarrier
# Uso:
#   make               # modo "fast"
#   make MODE=safe     # sem fast-math
#   make clean

CXX   := g++
MODE  ?= fast

# Flags comuns
COMMON_FLAGS := -std=c++17 -Wall -Wextra -Wno-unused-parameter

# Modo agressivo
FAST_FLAGS := -Ofast -flto -march=native -mtune=native \
              -ffast-math -fno-math-errno -funroll-loops \
              -fomit-frame-pointer -falign-functions=32 -falign-loops=32

# Modo "safe"
SAFE_FLAGS := -O3 -flto -march=native -mtune=native \
              -funroll-loops -fomit-frame-pointer

ifeq ($(MODE),fast)
  CXXFLAGS := $(COMMON_FLAGS) $(FAST_FLAGS)
else
  CXXFLAGS := $(COMMON_FLAGS) $(SAFE_FLAGS)
endif

# OpenMP para versões paralelas
OMPFLAGS := -fopenmp -pthread
LDLIBS   := -lm

# Alvos/bins
APPS := \
  hopscotch2d_serial \
  hopscotch2d_omp_naive \
  hopscotch2d_omp_semaphores \
  hopscotch2d_omp_sem_nobarrier \
  hopscotch2d_omp_busywait_barrier \
  hopscotch2d_omp_busywait_nobarrier

# Fontes (mesmo nome + .cpp)
SRCS := $(addsuffix .cpp,$(APPS))

.PHONY: all clean

all: $(APPS)

# Flags específicos por padrão de alvo:
# - serial sem OpenMP
hopscotch2d_serial: TFLAGS := $(CXXFLAGS)
# - qualquer alvo que comece com hopscotch2d_omp_ usa OpenMP
hopscotch2d_omp_%: TFLAGS := $(CXXFLAGS) $(OMPFLAGS)

# Regra geral: binário com mesmo nome do .cpp
%: %.cpp
	$(CXX) $(TFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f $(APPS) *.o
