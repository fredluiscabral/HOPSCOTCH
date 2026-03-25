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

# MPI puro
MPI_NAIVE_SRC   ?= hopscotch2d_mpi_naive.cpp
MPI_OVERLAP_SRC ?= hopscotch2d_mpi_overlap.cpp

# Híbridos MPI/OpenMP
HIB_NAIVE_SRC     ?= hopscotch2d_hib_naive.cpp
HIB_BUSYWAIT_SRC  ?= hopscotch2d_hib_busywait_nobarrier.cpp
HIB_SEMAPHORE_SRC ?= hopscotch2d_hib_sem_nobarrier.cpp
HIB_EWS_SRC       ?= hopscotch2d_hib_ews.cpp

# Apenas OpenMP
OMP_NAIVE_SRC             ?= hopscotch2d_omp_naive.cpp
OMP_NAIVE_NOFS_SRC        ?= hopscotch2d_omp_naive_nofs.cpp
OMP_BUSYWAIT_SRC          ?= hopscotch2d_omp_busywait_nobarrier.cpp
OMP_BUSYWAIT_NOFS_SRC     ?= hopscotch2d_omp_busywait_nobarrier_nofs.cpp
OMP_MPILIKE_SRC           ?= hopscotch2d_omp_mpilike.cpp
OMP_SEMAPHORE_SRC         ?= hopscotch2d_omp_sem_nobarrier.cpp
OMP_SEMAPHORE_NOFS_SRC    ?= hopscotch2d_omp_sem_nobarrier_nofs.cpp
OMP_EWS_SRC               ?= hopscotch2d_omp_ews.cpp

# Se tiver arquivos comuns C++ (sem main), liste separadamente:
COMMON_CPP_MPI ?=
COMMON_CPP_OMP ?=

COMMON_OBJS_MPI := $(patsubst %.cpp,$(BUILD_DIR)/%.mpi.o,$(COMMON_CPP_MPI))
COMMON_OBJS_OMP := $(patsubst %.cpp,$(BUILD_DIR)/%.omp.o,$(COMMON_CPP_OMP))

# Objetos MPI
MPI_NAIVE_OBJ     := $(BUILD_DIR)/$(MPI_NAIVE_SRC:.cpp=.mpi.o)
MPI_OVERLAP_OBJ   := $(BUILD_DIR)/$(MPI_OVERLAP_SRC:.cpp=.mpi.o)
HIB_NAIVE_OBJ     := $(BUILD_DIR)/$(HIB_NAIVE_SRC:.cpp=.mpi.o)
HIB_BUSYWAIT_OBJ  := $(BUILD_DIR)/$(HIB_BUSYWAIT_SRC:.cpp=.mpi.o)
HIB_SEMAPHORE_OBJ := $(BUILD_DIR)/$(HIB_SEMAPHORE_SRC:.cpp=.mpi.o)
HIB_EWS_OBJ       := $(BUILD_DIR)/$(HIB_EWS_SRC:.cpp=.mpi.o)

# Objetos OpenMP-only
OMP_NAIVE_OBJ          := $(BUILD_DIR)/$(OMP_NAIVE_SRC:.cpp=.omp.o)
OMP_NAIVE_NOFS_OBJ     := $(BUILD_DIR)/$(OMP_NAIVE_NOFS_SRC:.cpp=.omp.o)
OMP_BUSYWAIT_OBJ       := $(BUILD_DIR)/$(OMP_BUSYWAIT_SRC:.cpp=.omp.o)
OMP_BUSYWAIT_NOFS_OBJ  := $(BUILD_DIR)/$(OMP_BUSYWAIT_NOFS_SRC:.cpp=.omp.o)
OMP_MPILIKE_OBJ        := $(BUILD_DIR)/$(OMP_MPILIKE_SRC:.cpp=.omp.o)
OMP_SEMAPHORE_OBJ      := $(BUILD_DIR)/$(OMP_SEMAPHORE_SRC:.cpp=.omp.o)
OMP_SEMAPHORE_NOFS_OBJ := $(BUILD_DIR)/$(OMP_SEMAPHORE_NOFS_SRC:.cpp=.omp.o)
OMP_EWS_OBJ            := $(BUILD_DIR)/$(OMP_EWS_SRC:.cpp=.omp.o)

# Binários
MPI_BINS := \
	hopscotch2d_mpi_naive \
	hopscotch2d_mpi_overlap

HIB_BINS := \
	hopscotch2d_hib_naive \
	hopscotch2d_hib_busywait_nobarrier \
	hopscotch2d_hib_sem_nobarrier \
	hopscotch2d_hib_ews

OMP_BINS := \
	hopscotch2d_omp_naive \
	hopscotch2d_omp_naive_nofs \
	hopscotch2d_omp_busywait_nobarrier \
	hopscotch2d_omp_busywait_nobarrier_nofs \
	hopscotch2d_omp_mpilike \
	hopscotch2d_omp_sem_nobarrier \
	hopscotch2d_omp_sem_nobarrier_nofs \
	hopscotch2d_omp_ews

# ----------------------------
# Alvos
# ----------------------------
.PHONY: all mpi hib omp clean
all: mpi hib omp

mpi: $(MPI_BINS)
hib: $(HIB_BINS)
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
# Linkagem MPI puro
# ----------------------------
hopscotch2d_mpi_naive: $(COMMON_OBJS_MPI) $(MPI_NAIVE_OBJ)
	$(MPICXX) $(CXXFLAGS_MPI) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_mpi_overlap: $(COMMON_OBJS_MPI) $(MPI_OVERLAP_OBJ)
	$(MPICXX) $(CXXFLAGS_MPI) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# ----------------------------
# Linkagem híbrida MPI/OpenMP
# ----------------------------
hopscotch2d_hib_naive: $(COMMON_OBJS_MPI) $(HIB_NAIVE_OBJ)
	$(MPICXX) $(CXXFLAGS_MPI) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_hib_busywait_nobarrier: $(COMMON_OBJS_MPI) $(HIB_BUSYWAIT_OBJ)
	$(MPICXX) $(CXXFLAGS_MPI) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_hib_sem_nobarrier: $(COMMON_OBJS_MPI) $(HIB_SEMAPHORE_OBJ)
	$(MPICXX) $(CXXFLAGS_MPI) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_hib_ews: $(COMMON_OBJS_MPI) $(HIB_EWS_OBJ)
	$(MPICXX) $(CXXFLAGS_MPI) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# ----------------------------
# Linkagem OpenMP-only
# ----------------------------
hopscotch2d_omp_naive: $(COMMON_OBJS_OMP) $(OMP_NAIVE_OBJ)
	$(CXX) $(CXXFLAGS_OMP) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_omp_naive_nofs: $(COMMON_OBJS_OMP) $(OMP_NAIVE_NOFS_OBJ)
	$(CXX) $(CXXFLAGS_OMP) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_omp_busywait_nobarrier: $(COMMON_OBJS_OMP) $(OMP_BUSYWAIT_OBJ)
	$(CXX) $(CXXFLAGS_OMP) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_omp_busywait_nobarrier_nofs: $(COMMON_OBJS_OMP) $(OMP_BUSYWAIT_NOFS_OBJ)
	$(CXX) $(CXXFLAGS_OMP) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_omp_mpilike: $(COMMON_OBJS_OMP) $(OMP_MPILIKE_OBJ)
	$(CXX) $(CXXFLAGS_OMP) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_omp_sem_nobarrier: $(COMMON_OBJS_OMP) $(OMP_SEMAPHORE_OBJ)
	$(CXX) $(CXXFLAGS_OMP) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_omp_sem_nobarrier_nofs: $(COMMON_OBJS_OMP) $(OMP_SEMAPHORE_NOFS_OBJ)
	$(CXX) $(CXXFLAGS_OMP) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_omp_ews: $(COMMON_OBJS_OMP) $(OMP_EWS_OBJ)
	$(CXX) $(CXXFLAGS_OMP) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# ----------------------------
# Limpeza
# ----------------------------
clean:
	rm -rf $(BUILD_DIR) $(MPI_BINS) $(HIB_BINS) $(OMP_BINS)