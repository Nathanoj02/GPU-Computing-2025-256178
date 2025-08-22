# =============================================================================
# OpenCV C++ Makefile - Compile from src/ to bin/
# =============================================================================

# Compiler and flags
CXX = g++
CC = gcc
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
CFLAGS = -std=c99 -Wall -Wextra -O2 -g

# OpenCV flags
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

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
	mkdir -p $(BINDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

# Rule to compile .cpp to .o (object files)
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling C++: $<..."
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -I $(INCDIR) -c $< -o $@

# Rule to compile .c to .o (object files)
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@echo "Compiling C: $<..."
	$(CC) $(CFLAGS) -I $(INCDIR) -c $< -o $@

# Rule to link object files to executables
$(BINDIR)/%: $(OBJDIR)/%.o $(C_OBJECTS) | $(BINDIR)
	@echo "Linking $@..."
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LIBS) -lm

# Clean generated files
clean:
	@echo "Cleaning..."
	rm -rf $(BINDIR) $(OBJDIR)

# Clean and rebuild
rebuild: clean all

# Show help
help:
	@echo "Available targets:"
	@echo "  all      - Build all executables (default)"
	@echo "  clean    - Remove generated files"
	@echo "  rebuild  - Clean and build all"
	@echo "  help     - Show this help"
	@echo ""
	@echo "Structure:"
	@echo "  src/     - Source .cpp and .c files"
	@echo "  inc/     - Header files"
	@echo "  bin/     - Generated executables"
	@echo "  obj/     - Generated object files"

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
	@echo "OPENCV_CFLAGS: $(OPENCV_CFLAGS)"
	@echo "OPENCV_LIBS: $(OPENCV_LIBS)"
	@echo "CPP_SOURCES: $(CPP_SOURCES)"
	@echo "C_SOURCES: $(C_SOURCES)"
	@echo "ALL_OBJECTS: $(ALL_OBJECTS)"
	@echo "EXECUTABLES: $(EXECUTABLES)"

# Phony targets
.PHONY: all clean rebuild help list debug