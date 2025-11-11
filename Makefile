# =============================================================================
# OpenCV C++ Makefile - Compile from src/ to bin/
# =============================================================================

# Detect GPU compute capability
GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

# Compiler and flags
CXX = nvc++
CC = nvc
CXXFLAGS = -std=c++17 -Wall -O2 -g -acc -gpu=cc$(GPU_CC) --diag_suppress partial_override
CFLAGS = -std=c99 -Wall -O2 -g -acc -gpu=cc$(GPU_CC)

# Profiling flags for Nsight Systems
PROFILE_CXXFLAGS = -std=c++17 -O2 -g -lineinfo -acc -gpu=cc$(GPU_CC) --diag_suppress partial_override
PROFILE_CFLAGS = -std=c99 -O2 -g -lineinfo -acc -gpu=cc$(GPU_CC)
PROFILE_LINKFLAGS = -std=c++17 -O2 -g -acc -gpu=cc$(GPU_CC) --diag_suppress partial_override

# OpenCV flags
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

GCC_LIB_PATH = /opt/shares/easybuild/software/GCCcore/12.3.0/lib64

LDFLAGS = -Wl,-z,noexecstack \
          -L$(GCC_LIB_PATH) \
          -Wl,-rpath,$(GCC_LIB_PATH)

# Directories
SRCDIR = src
INCDIR = inc
BINDIR = bin
OBJDIR = obj

# Find source files
CPP_SOURCES = $(wildcard $(SRCDIR)/*.cpp)
C_SOURCES = $(wildcard $(SRCDIR)/*.c)

# Generate object file names
CPP_OBJECTS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES))
C_OBJECTS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(C_SOURCES))
ALL_OBJECTS = $(CPP_OBJECTS) $(C_OBJECTS)

# Generate executable names (from .cpp files only)
EXECUTABLES = $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%,$(CPP_SOURCES))

# Default target - build all executables
all: $(EXECUTABLES)

# Create directories if they don't exist
$(BINDIR):
	@mkdir -p $(BINDIR)

$(OBJDIR):
	@mkdir -p $(OBJDIR)

# Rule to compile .cpp to .o (object files)
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling C++: $<..."
	@$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -I $(INCDIR) -c $< -o $@

# Rule to compile .c to .o (object files)
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@echo "Compiling C: $<..."
	@$(CC) $(CFLAGS) -I $(INCDIR) -c $< -o $@

# Rule to link object files to executables
$(BINDIR)/%: $(OBJDIR)/%.o $(C_OBJECTS) | $(BINDIR)
	@echo "Linking $@..."
	@$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(OPENCV_LIBS) -lm
	@echo "✓ Build complete: $@"

# Special rule for profiling build of main_profile
profile: $(BINDIR)/main_profile

# Profiling object files with special flags
$(OBJDIR)/main_profile_prof.o: $(SRCDIR)/main_profile.cpp | $(OBJDIR)
	@echo "Compiling C++ for profiling: $<..."
	@$(CXX) $(PROFILE_CXXFLAGS) $(OPENCV_CFLAGS) -I $(INCDIR) -c $< -o $@

$(OBJDIR)/%_prof.o: $(SRCDIR)/%.c | $(OBJDIR)
	@echo "Compiling C for profiling: $<..."
	@$(CC) $(PROFILE_CFLAGS) -I $(INCDIR) -c $< -o $@

# Generate profiling C objects from regular C sources
C_OBJECTS_PROFILE = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%_prof.o,$(C_SOURCES))

# Link main_profile with profiling flags
$(BINDIR)/main_profile: $(OBJDIR)/main_profile_prof.o $(C_OBJECTS_PROFILE) | $(BINDIR)
	@echo "Linking $@ for profiling..."
	@$(CXX) $(PROFILE_LINKFLAGS) $^ -o $@ $(LDFLAGS) $(OPENCV_LIBS) -lm
	@echo "✓ Profiling build complete: $@"
	@echo "Run with: nsys profile $(BINDIR)/main_profile [args]"

# Clean generated files
clean:
	@echo "Cleaning..."
	@rm -rf $(BINDIR) $(OBJDIR)
	@echo "✓ Clean complete"

# Clean and rebuild
rebuild: clean all

# Show help
help:
	@echo "Available targets:"
	@echo "  all      - Build all executables (default)"
	@echo "  ncu      - Build main_profile with profiling flags for Nsight Compute"
	@echo "  clean    - Remove generated files"
	@echo "  rebuild  - Clean and build all"
	@echo "  help     - Show this help"
	@echo ""
	@echo "Structure:"
	@echo "  src/     - Source .cpp and .c files"
	@echo "  inc/     - Header files"
	@echo "  bin/     - Generated executables"
	@echo "  obj/     - Generated object files"
	@echo ""
	@echo "NCU Profiling:"
	@echo "  make ncu                    - Build main_profile for profiling"
	@echo "  ncu --set full bin/main_profile_ncu [args] - Profile with Nsight Compute"

# Show found source files
list:
	@echo "C++ source files found:"
	@for file in $(CPP_SOURCES); do echo "  $$file"; done
	@echo ""
	@echo "C source files found:"
	@for file in $(C_SOURCES); do echo "  $$file"; done
	@echo ""
	@echo "Will generate executables:"
	@for file in $(EXECUTABLES); do echo "  $$file"; done

# Debug info
debug:
	@echo "CXX: $(CXX)"
	@echo "CC: $(CC)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "PROFILE_CXXFLAGS: $(PROFILE_CXXFLAGS)"
	@echo "PROFILE_CFLAGS: $(PROFILE_CFLAGS)"
	@echo "OPENCV_CFLAGS: $(OPENCV_CFLAGS)"
	@echo "OPENCV_LIBS: $(OPENCV_LIBS)"
	@echo "CPP_SOURCES: $(CPP_SOURCES)"
	@echo "C_SOURCES: $(C_SOURCES)"
	@echo "ALL_OBJECTS: $(ALL_OBJECTS)"
	@echo "EXECUTABLES: $(EXECUTABLES)"

# Phony targets
.PHONY: all clean rebuild help list debug profile