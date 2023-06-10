.PHONY: all, clean, init, ground_truth

CC = clang


all:
	@echo Done!


clean:


clear: clean
	rm -rf openblas
	rm -rf OpenBLAS*


init:
#	copy init makefiles from attack/source/llvm/
	cp -r ../llvm/openblas_makefiles/init/ ./openblas_makefiles/
	cp -r ../llvm/openblas_makefiles/fast/ ./openblas_makefiles/
# 	unzip openblas source
	tar -xzf ../../../poc/openblas.tar.gz -C ./
	mv OpenBLAS* openblas
#	compile openblas with clang (first-time compile)
	cp ./openblas_makefiles/init/* openblas/
	make -C openblas/
# 	after first-time compile, copy fast-compile Makefiles
	cp ./openblas_makefiles/fast/* openblas/
	@echo Done!


# identify a experiment configuration
EXP_CONF = gtsrb_vgg16
EXP_DATASET = gtsrb
EXP_NETWORK = vgg16
# - gtsrb: ResNet34( GTSRB )
# - imagenet_resnet: ResNet50( ImageNet )
# - imagenet_vgg: VGG-16( ImageNet )
# - fmnist: LeNet( FashionMNIST )

# specify a branch index instance to evaluate
EXP_IDX = -1
# -1 evaluates all instances

# get branch flipping ground truth data 
ground_truth: llvm-pass/build/flipLauncher/libflipLauncher.so
#	copy attacker's Makefile
	cp -r ./openblas_makefiles/attack/* ./openblas/
#	reset the branch index counter file (just remove it)
	rm -f br_info.tmp
#	apply flipbranches pass to object files
	rm -f ./openblas/interface/cblas_dgemm.bc
	rm -f ./openblas/driver/level3/dgemm_nn.bc
	rm -f ./openblas/driver/level3/dgemm_nt.bc
	rm -f ./openblas/driver/level3/dgemm_tn.bc
	rm -f ./openblas/driver/level3/dgemm_tt.bc
	make -C ./openblas/interface/ cblas_dgemm.bc
	make -C ./openblas/driver/level3/ dgemm_nn.bc
	make -C ./openblas/driver/level3/ dgemm_nt.bc
	make -C ./openblas/driver/level3/ dgemm_tn.bc
	make -C ./openblas/driver/level3/ dgemm_tt.bc
#	evaluate the clean instance, as a reference example for processing results
	@if ! [ -e ./results/${EXP_CONF}/clean/poc.out ]; then \
		mkdir -p ./results/${EXP_CONF}/clean/; \
		make -C ../../../poc/infras GR_EXP_CONF=${EXP_CONF} gr-clean-instance; \
	fi;
	@if ! [ -e ./results/${EXP_CONF}/clean/done ]; then \
		echo running GR clean instance ...; \
		export LD_LIBRARY_PATH=$$ld_library_orig:${PWD}/../../../poc/infras/openblas/lib/; \
		echo LD_LIBRARY_PATH=$$LD_LIBRARY_PATH; \
		./results/${EXP_CONF}/clean/poc.out ${EXP_DATASET} ${EXP_NETWORK} > ./results/${EXP_CONF}/clean/log.txt 2>&1; \
		touch ./results/${EXP_CONF}/clean/done; \
	fi;
#	iterate over branches of interest with a bash-style for loop (on Ubuntu 20.04)
#	Reference: https://askubuntu.com/questions/621981/for-loop-syntax-in-shell-script
#	TODO: Note that you may need another way to iterate over a range for a different OS
#	The value of `br_lim` is taken from the output of a python script
#	Reference: https://stackoverflow.com/questions/34171568/return-value-from-python-script-to-shell-script
	@br_lim=`python br_info_reader.py` ; \
	ld_library_orig=$$LD_LIBRARY_PATH ; \
	for br_idx in $$(seq 0 $$br_lim) ; \
		do echo br_idx = $$br_idx; \
		if ! [ -e ./openblas_inst/br$$br_idx/lib/libopenblas.so ]; then \
			mkdir -p ./openblas_inst/br$$br_idx/; \
			rm -f ./openblas/interface/cblas_dgemm.o; \
			rm -f ./openblas/driver/level3/dgemm_nn.o; \
			rm -f ./openblas/driver/level3/dgemm_nt.o; \
			rm -f ./openblas/driver/level3/dgemm_tn.o; \
			rm -f ./openblas/driver/level3/dgemm_tt.o; \
			make -s -C ./openblas BR_IDX=$$br_idx; \
			make -s -C ./openblas PREFIX=${PWD}/openblas_inst/br$$br_idx install; \
		fi; \
		if [ ${EXP_IDX} -eq -1 ] || [ $$br_idx -eq ${EXP_IDX} ]; then \
			if ! [ -e ./results/${EXP_CONF}/br$$br_idx/poc.out ]; then \
				mkdir -p ./results/${EXP_CONF}/br$$br_idx/; \
				make -C ../../../poc/infras GR_EXP_CONF=${EXP_CONF} GR_BR_IDX=$$br_idx gr-instance; \
			fi; \
			if ! [ -e ./results/${EXP_CONF}/br$$br_idx/done ]; then \
				echo running GR instance $$br_idx ...; \
				export LD_LIBRARY_PATH=$$ld_library_orig:${PWD}/openblas_inst/br$$br_idx/lib:/usr/local/lib ; \
				echo LD_LIBRARY_PATH=$$LD_LIBRARY_PATH; \
				timeout -v 30m ./results/${EXP_CONF}/br$$br_idx/poc.out ${EXP_DATASET} ${EXP_NETWORK} > ./results/${EXP_CONF}/br$$br_idx/log.txt 2>&1; \
				touch ./results/${EXP_CONF}/br$$br_idx/done; \
			fi; \
		fi; \
	done
	@echo Done!

