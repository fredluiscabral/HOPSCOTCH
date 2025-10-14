# Makefile — compila versões: serial, OpenMP (naive/semáforos/busy-wait) e híbrida MPI+OpenMP
# Uso:
#   make            # modo "fast"
#   make MODE=safe  # modo "safe" (sem fast-math)
#   make clean

CXX    ?= g++
MPICXX ?= mpicxx
MODE   ?= fast

# Flags comuns (silencia -Wcomment por causa de comentários com '\')
COMMON_FLAGS := -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-comment

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

# OpenMP e MPI
OMPFLAGS   := -fopenmp -pthread
# Evita incluir os C++ bindings obsoletos do MPI (elimina aqueles warnings chatos)
MPI_DEFS   := -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1

LDLIBS := -lm

# Alvos
APPS := \
  hopscotch2d_serial \
  hopscotch2d_omp_naive \
  hopscotch2d_omp_semaphores \
  hopscotch2d_omp_sem_nobarrier \
  hopscotch2d_omp_busywait_barrier \
  hopscotch2d_omp_busywait_nobarrier \
  hopscotch2d_hib_naive

.PHONY: all clean
all: $(APPS)

# Flags por alvo:
hopscotch2d_serial:      TFLAGS := $(CXXFLAGS)
hopscotch2d_omp_%:       TFLAGS := $(CXXFLAGS) $(OMPFLAGS)

# Regra genérica (serial e OpenMP)
%: %.cpp
	$(CXX) $(TFLAGS) $< -o $@ $(LDLIBS)

# Regra específica para a versão híbrida (MPI+OpenMP)
hopscotch2d_hib_naive: hopscotch2d_hib_naive.cpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(MPI_DEFS) $< -o $@ $(LDLIBS)

clean:
	rm -f $(APPS) *.o
