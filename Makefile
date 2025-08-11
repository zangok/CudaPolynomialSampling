# Compiler and flags
NVCC = nvcc
CFLAGS = -O2 -rdc=true 

# Target
TARGET = ProjectCuda3.exe

# Source files
CU_SRCS = ProjectCuda3.cu utils.cu PolynomialSampling.cu ExperimentManager.cu InflectionFinder.cu CountPositives.cu

OBJS = ProjectCuda3.o utils.o PolynomialSampling.o ExperimentManager.o InflectionFinder.o CountPositives.o

# Build all
all: $(TARGET)

# Link object files into executable
$(TARGET): $(OBJS)
	$(NVCC) -o $@ $^

# Compile CUDA source files to object files
%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	del /Q $(TARGET) *.o
