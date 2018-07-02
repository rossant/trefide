# Detect Operating System
ifeq ($(OS),Windows_NT)
    detected_OS := Windows
else
    detected_OS := $(shell sh -c 'uname -s 2>/dev/null || echo not')
endif

# Set Windows Specific Environment Variables: TODO
ifeq ($(detected_OS),Windows)
    echo "Installation on Windows not currently supported."
endif

# Set MacOS Specific Environment Variables
ifeq ($(detected_OS),Darwin)
    EXT=.dylib
    LDFLAGS = -dynamiclib
endif

# Set Linux Specific Environment Variables
ifeq ($(detected_OS),Linux)
    EXT=.so
    LDFLAGS = -shared
endif

# TODO: Set Compiler Specific Environment Variables

# OS Agnostic Environment Variables
CXX = icpc
CXXFLAGS = -mkl=sequential -qopenmp -O3 -fPIC -I$(shell pwd)/proxtv/  -L$(shell pwd)/$(PROXTV) -I$(shell pwd)/glmgen/include/ -L$(shell pwd)/$(GLMGEN)/lib -D NOMATLAB=1 #-mkl=parallel
PROXTV = proxtv
LIBPROXTV = $(PROXTV)/libproxtv$(EXT)
GLMGEN = glmgen
LIBGLMGEN = $(GLMGEN)/lib/libglmgen$(EXT)
LIBTREFIDE = libtrefide$(EXT)
LIBS = -lproxtv -lglmgen -lmkl_intel_lp64 -lmkl_core -lm -lmkl_intel_thread -liomp5
SRCS = utils/welch.cpp proxtf/wpdas.cpp proxtf/line_search.cpp proxtf/utils.cpp proxtf/l1tf/ipm.cpp proxtf/admm.cpp pmd/pmd.cpp pmd/decimation.cpp
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

.PHONY: all
all: $(LIBPROXTV) $(LIBGLMGEN) $(LIBTREFIDE)

$(LIBPROXTV):
	cd $(PROXTV); make;

$(LIBGLMGEN):
	cd $(GLMGEN); make;

$(LIBTREFIDE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

$(SRCS:.cpp=.d) : %.d : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY : clean 
clean:
	rm -f $(LIBTREFIDE) $(OBJS) $(SRCS:.cpp=.d)
	cd $(PROXTV); make clean;
	cd $(GLMGEN); make clean;
