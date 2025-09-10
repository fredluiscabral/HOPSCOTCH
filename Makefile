# Makefile — compila versões serial, naive, semáforos e busy-wait (com/sem barreiras)
# Uso:
#   make                  # compila tudo no modo "fast"
#   make MODE=safe        # compila tudo no modo "safe" (sem fast-math)
#   make clean

CXX := g++
MODE ?= fast

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

# OpenMP + pthread para versões paralelas
OMPFLAGS := -fopenmp -pthread
LDLIBS := -lm

# Alvos
SERIAL_APP      := hopscotch2d_serial
NAIVE_APP       := hopscotch2d_omp_naive
SEM_NOBAR_APP   := hopscotch2d_omp_sem_nobarrier
SEM_BARRIER     := hopscotch2d_omp_semaphores
BW_BAR_APP      := hopscotch2d_omp_busywait_barrier
BW_NOBAR_APP    := hopscotch2d_omp_busywait_nobarrier

# Fontes (ajuste os que você tiver no diretório)
SERIAL_SRC    := hopscotch2d_serial.cpp
NAIVE_SRC     := hopscotch2d_omp_naive.cpp
SEM_NOBAR_SRC := hopscotch2d_omp_sem_nobarrier.cpp
SEM_BARRIER   := hopscotch2d_omp_semaphores.cpp
BW_BAR_SRC    := hopscotch2d_omp_busywait_barrier.cpp
BW_NOBAR_SRC  := hopscotch2d_omp_busywait_nobarrier.cpp

.PHONY: all clean

all: $(SERIAL_APP) $(NAIVE_APP) $(SEM_NOBAR_APP) $(BW_BAR_APP) $(BW_NOBAR_APP)

$(SERIAL_APP): $(SERIAL_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

$(NAIVE_APP): $(NAIVE_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDLIBS)

$(SEM_NOBAR_APP): $(SEM_NOBAR_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDLIBS)

$(BW_BAR_APP): $(BW_BAR_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDLIBS)

$(BW_NOBAR_APP): $(BW_NOBAR_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f $(SERIAL_APP) $(NAIVE_APP) $(SEM_NOBAR_APP) $(BW_BAR_APP) $(BW_NOBAR_APP) *.o
