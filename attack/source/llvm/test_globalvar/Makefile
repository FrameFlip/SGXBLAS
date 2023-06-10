.PHONY: all

CXX = clang++


all: main.out
	@export LD_LIBRARY_PATH=${HOME}/Documents/projects/sgxblas/attack/source/llvm/test_globalvar${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}; \
	./main.out


clean:
	rm -f *.ll *.bc *.o *.so *.out
	rm -f ../br_idx.tmp


func.o: func.bc
	opt -enable-new-pm=0 -load ../llvm-pass/build/countBranches/libcountBranches.so -countbranches -o func_inst.bc func.bc
	$(CXX) -c -fPIC -Wall -Werror -o $@ func_inst.bc

func.bc: func.cc
	clang++ -c -emit-llvm -fno-discard-value-names -o $@ $<

libfunc.so: func.o
	$(CXX) -shared -fpic -Wl,-soname,libfunc.so -o $@ $< -lc


main.o: main.cc
	$(CXX) -c -o $@ $<


main.out: main.o libfunc.so
	$(CXX) -Wall -L. -o $@ $< -lfunc
	
