MPICXX ?= mpicxx
BUILD_DIR ?= build

OPT ?= -O3
DBG ?= -g -fno-omit-frame-pointer -fno-optimize-sibling-calls
OPENMP ?= -fopenmp
WARN ?= -Wall -Wextra -Wshadow -Wundef -Wno-unused-parameter
DEFS ?= -D_GNU_SOURCE

CXXFLAGS ?= -std=c++17 $(WARN) $(OPT) $(DBG) $(OPENMP) $(DEFS)
LDFLAGS ?=
LDLIBS ?= -lm -pthread

# Se tiver arquivos comuns C++ (sem main), liste aqui:
COMMON_CPP ?=
COMMON_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(COMMON_CPP))

NAIVE_SRC     ?= hopscotch2d_hib_naive.cpp
BUSYWAIT_SRC  ?= hopscotch2d_hib_busywait_nobarrier.cpp
SEMAPHORE_SRC ?= hopscotch2d_hib_sem_nobarrier.cpp

NAIVE_OBJ     := $(BUILD_DIR)/hopscotch2d_hib_naive.o
BUSYWAIT_OBJ  := $(BUILD_DIR)/hopscotch2d_hib_busywait_nobarrier.o
SEMAPHORE_OBJ := $(BUILD_DIR)/hopscotch2d_hib_sem_nobarrier.o

.PHONY: all clean
all: hopscotch2d_hib_naive hopscotch2d_hib_busywait_nobarrier hopscotch2d_hib_sem_nobarrier

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

hopscotch2d_hib_naive: $(COMMON_OBJS) $(NAIVE_OBJ)
	$(MPICXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_hib_busywait_nobarrier: $(COMMON_OBJS) $(BUSYWAIT_OBJ)
	$(MPICXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

hopscotch2d_hib_sem_nobarrier: $(COMMON_OBJS) $(SEMAPHORE_OBJ)
	$(MPICXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -rf $(BUILD_DIR) hopscotch2d_hib_naive hopscotch2d_hib_busywait_nobarrier hopscotch2d_hib_sem_nobarrier
