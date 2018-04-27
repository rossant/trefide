CXX = icpc
CXXFLAGS = -mkl=sequential -qopenmp -O3 -fPIC -I$(shell pwd)/proxtv -L$(shell pwd)/proxtv -D NOMATLAB=1 #-mkl=parallel
LDFLAGS = -dynamiclib
PROXTV = proxtv
LIBPROXTV = $(PROXTV)/libproxtv.dylib
GLMGEN = glmgen
LIBGLMGEN = $(GLMGEN)/libglmgen.dylib
TARGET_LIB = libtrefide.dylib
LIBS = -lmkl_intel_lp64 -lmkl_core -lm -lmkl_intel_thread -liomp5 -lproxtv
SRCS = utils/welch.cpp proxtf/wpdas.cpp proxtf/line_search.cpp proxtf/utils.cpp proxtf/l1tf/ipm.cpp pmd/pmd.cpp
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

.PHONY: all
all: $(LIBPROXTV) $(LIBGLMGEN) $(TARGET_LIB)

$(LIBPROXTV):
	cd $(PROXTV); make;

$(LIBGLMGEN):
	cd $(GLMGEN); make;

$(TARGET_LIB): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

$(SRCS:.cpp=.d) : %.d : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY : clean 
clean:
	rm -f $(TARGET_LIB) $(OBJS) $(SRCS:.cpp=.d)
	cd $(PROXTV); make clean;
	cd $(GLMGEN); make clean;
