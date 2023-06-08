.PHONY: all

SHELL := /bin/bash

# Changing openblas version is not recommended. Unexpected bugs may occur.
OPENBLAS = openblas
# You may check https://github.com/xianyi/OpenBLAS/releases for latest
OPENBLAS_VERSION=0.3.20
OPENBLAS_SOURCE = https://github.com/xianyi/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz


all: openblas
# 	download traffic data
	make -C ./data/traffic/
	make -C ./infras/ poc.out
	@echo Done!


clean:
	rm -f ${OPENBLAS}.tar.gz

clear: clean
	rm -rf openblas/
	rm -rf infras/openblas/


openblas: 
	@echo Please make sure you have installed make, cmake, gcc, g++, gfortran, clang, and opencv
	sudo apt-get -y install make cmake gcc g++ gfortran clang libopencv-dev python3-opencv
	curl -o ${OPENBLAS}.tar.gz -L ${OPENBLAS_SOURCE}
	tar -xzvf ${OPENBLAS}.tar.gz
	mv OpenBLAS* openblas
	make -C openblas/
#	make -C openblas/ PREFIX=/usr/lib/openblas install # needs sudo
	make -C openblas/ PREFIX=../infras/openblas install
	@echo openblas v${OPENBLAS_VERSION} installation done 