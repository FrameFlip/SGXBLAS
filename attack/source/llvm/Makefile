.PHONY: all, clean, init, inst_openblas


all:
	@echo Done!


clean:


clear: clean
	rm -rf openblas
	rm -rf OpenBLAS*


init:
# 	unzip openblas source
	tar -xzf ../../../poc/openblas.tar.gz -C ./
	mv OpenBLAS* openblas
#	compile openblas with clang (first-time compile)
	cp ./openblas_makefiles/init/* openblas/
	make -C openblas/
# 	after first-time compile, copy fast-compile Makefiles
	cp ./openblas_makefiles/fast/* openblas/
	@echo Done!


# apply LLVM-based instrumentation to openblas
inst_openblas: llvm-pass/build/countBranches/libcountBranches.so
#	copy Makefiles that apply the LLVM Pass
	cp -r ./openblas_makefiles/attack/* ./openblas/
#	remove object files that need to be instrumented
	rm -f ./openblas/interface/cblas_dgemm.o
	rm -f ./openblas/driver/level3/dgemm_nn.o
	rm -f ./openblas/driver/level3/dgemm_nt.o
	rm -f ./openblas/driver/level3/dgemm_tn.o
	rm -f ./openblas/driver/level3/dgemm_tt.o
#	reset the branch index counter file (just remove it)
	rm -f br_idx.tmp
#	re-compile openblas using the attack Makefiles
	mkdir -p ./openblas_inst
	make -C ./openblas
	make -C ./openblas PREFIX=${PWD}/openblas_inst install

