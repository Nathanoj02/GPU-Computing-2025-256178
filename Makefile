# =============================================================================
# OpenCV C++ Makefile - Compile from src/ to bin/
# =============================================================================

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g

# OpenCV flags
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# Directories
SRCDIR = src
BINDIR = bin
OBJDIR = obj

# Find all .cpp files in src directory
SOURCES = $(wildcard $(SRCDIR)/*.cpp)

# Generate executable names (remove .cpp extension and src/ path)
EXECUTABLES = $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%,$(SOURCES))

# Generate object file names
OBJECTS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))

# Default target - build all executables
all: $(EXECUTABLES)

# Create directories if they don't exist
$(BINDIR):
	mkdir -p $(BINDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

# Rule to compile .cpp to .o (object files)
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $@

# Rule to link object files to executables
$(BINDIR)/%: $(OBJDIR)/%.o | $(BINDIR)
	@echo "Linking $@..."
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV_LIBS)

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
	@echo "  src/     - Source .cpp files"
	@echo "  bin/     - Generated executables"
	@echo "  obj/     - Generated object files"

# Show found source files
list:
	@echo "Source files found:"
	@for file in $(SOURCES); do echo "  $$file"; done
	@echo ""
	@echo "Will generate executables:"
	@for file in $(EXECUTABLES); do echo "  $$file"; done

# Compile specific file (usage: make filename)
# This allows you to do: make test (to compile src/test.cpp to bin/test)
%: $(SRCDIR)/%.cpp | $(BINDIR) $(OBJDIR)
	@echo "Compiling single file: $<"
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c $< -o $(OBJDIR)/$@.o
	$(CXX) $(CXXFLAGS) $(OBJDIR)/$@.o -o $(BINDIR)/$@ $(OPENCV_LIBS)

# Run specific executable (usage: make run FILE=filename)
run:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make run FILE=filename"; \
		echo "Example: make run FILE=test"; \
	else \
		if [ -f "$(BINDIR)/$(FILE)" ]; then \
			echo "Running $(BINDIR)/$(FILE)..."; \
			./$(BINDIR)/$(FILE); \
		else \
			echo "Error: $(BINDIR)/$(FILE) not found. Compile it first with: make $(FILE)"; \
		fi \
	fi

# Debug info
debug:
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "OPENCV_CFLAGS: $(OPENCV_CFLAGS)"
	@echo "OPENCV_LIBS: $(OPENCV_LIBS)"
	@echo "SOURCES: $(SOURCES)"
	@echo "EXECUTABLES: $(EXECUTABLES)"
	@echo "OBJECTS: $(OBJECTS)"

# Phony targets
.PHONY: all clean rebuild help list run debug