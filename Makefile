# ----------------------------
# Compiladores
# ----------------------------
CXX    ?= g++
MPICXX ?= mpicxx

BUILD_DIR ?= build

# ----------------------------
# Flags
# ----------------------------
OPT    ?= -O3
DBG    ?= -g -fno-omit-frame-pointer -fno-optimize-sibling-calls
OPENMP ?= -fopenmp

# (Opcional) -Wno-comment silencia "multi-line comment" causado por "\" em // ...
WARN   ?= -Wall -Wextra -Wshadow -Wundef -Wno-unused-parameter -Wno-comment

# Definições gerais
DEFS   ?= -D_GNU_SOURCE

# IMPORTANTÍSSIMO: evita os headers MPI C++ antigos (OpenMPI/MPICH),
# que geram warnings chatos (cast-function-type etc.)
MPI_SKIP_CXX_BINDINGS ?= -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1

# CXXFLAGS por modo
CXXFLAGS_COMMON ?= -std=c++17 $(WARN) $(OPT) $(DBG) $(OPENMP)
CXXFLAGS_MPI    ?= $(CXXFLAGS_COMMON) $(DEFS) $(MPI_SKIP_CXX_BINDINGS)
CXXFLAGS_OMP    ?= $(CXXFLAGS_COMMON) $(DEFS)

LDFLAGS ?=
LDLIBS  ?= -lm -pthread

# ----------------------------
# Fontes (ajuste se necessário)
# ----------------------------

# Híbridos MPI/OpenMP
HIB_NAIVE_SRC     ?= hopscotch2d_hib_naive.cpp
HIB_BUSYWAIT_SRC  ?= hopscotch2d_hib_busywait_nobarrier.cpp
HIB_SEMAPHORE_SRC ?= hopscotch2d_hib_sem_nobarrier.cpp

# Apenas OpenMP (AJUSTE se os nomes forem diferentes no seu repo)
OMP_NAIVE_SRC     ?= hopscotch2d_omp_naive.cpp
OMP_BUSYWAIT_SRC  ?= hopscotch2d_omp_busywait_nobarrier.cpp
OMP_SEMAPHORE_SRC ?= hopscotch2d_omp_sem_nobarrier.cpp

# Se tiver arquivos comuns C++ (sem main), liste separadamente:
COMMON_CPP_MPI ?=
COMMON_CPP_OMP ?=

COMMON_OBJS_MPI := $(patsubst %.cpp,$(BUILD_DIR)/%.mpi.o,$(COMMON_CPP_MPI))
COMMON_OBJS_OMP := $(patsubst %.cpp,$(BUILD_DIR)/%.omp.o,$(COMMON_CPP_OMP))

# Objetos MPI
HIB_NAIVE_OBJ     := $(BUILD_DIR)/$(HIB_NAIVE_SRC:.cpp=.mpi.o)
HIB_BUSYWAIT_OBJ  := $(BUILD_DIR)/$(HIB_BUSYWAIT_SRC:.cpp=.mpi.o)
HIB_SEMAPHORE_OBJ := $(BUILD_DIR)/$(HIB_SEMAPHORE_SRC:.cpp=.mpi.o)

# Objetos OpenMP-only
OMP_NAIVE_OBJ     := $(BUILD_DIR)/$(OMP_NAIVE_SRC:.cpp=.omp.o)
OMP_BUSYWAIT_OBJ  := $(BUILD_DIR)/$(OMP_BUSYWAIT_SRC:.cpp=.omp.o)
OMP_SEMAPHORE_OBJ := $(BUILD_DIR)/$(OMP_SEMAPHORE_SRC:.cpp=.omp.o)

# Binários
HIB_BINS := hopscotch2d_hib_naive hopscotch2d_hib_busywait_nobarrier hopscotch2d_hib_sem_nobarrier
OMP_BINS := hopscotch2d_omp_naive hopscotch2d_omp_busywait_nobarrier hopscotch2d_omp_sem_nobarrier

# ----------------------------
# Alvos
# ----------------------------
.PHONY: all mpi omp clean
all: mpi omp
mpi: $(HIB_BINS)
omp: $(OMP_BINS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ----------------------------
# Regras de compilação
# ----------------------------

# Compila objetos MPI (mpicxx)
$(BUILD_DIR)/%.mpi.o: %.cpp | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS_MPI) -c $< -o $@

# Compila objetos OpenMP-only (g++)
$(BUILD_DIR)/%.omp.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_OMP) -c $< -o $@

# ----------------------------
# Linkagem MPI
# ----------------------------
hopscotch2d_hib_naive: $(COMMON_OBJS_MPI) $(HIB_NAIVE_OBJ)
	$(MPICXX) $(CXXFLAGS_MPI) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_hib_busywait_nobarrier: $(COMMON_OBJS_MPI) $(HIB_BUSYWAIT_OBJ)
	$(MPICXX) $(CXXFLAGS_MPI) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_hib_sem_nobarrier: $(COMMON_OBJS_MPI) $(HIB_SEMAPHORE_OBJ)
	$(MPICXX) $(CXXFLAGS_MPI) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# ----------------------------
# Linkagem OpenMP-only
# ----------------------------
hopscotch2d_omp_naive: $(COMMON_OBJS_OMP) $(OMP_NAIVE_OBJ)
	$(CXX) $(CXXFLAGS_OMP) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_omp_busywait_nobarrier: $(COMMON_OBJS_OMP) $(OMP_BUSYWAIT_OBJ)
	$(CXX) $(CXXFLAGS_OMP) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_omp_sem_nobarrier: $(COMMON_OBJS_OMP) $(OMP_SEMAPHORE_OBJ)
	$(CXX) $(CXXFLAGS_OMP) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# ----------------------------
# Limpeza
# ----------------------------
clean:
	rm -rf $(BUILD_DIR) $(HIB_BINS) $(OMP_BINS)
