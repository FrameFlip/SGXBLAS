#!/bin/bash
.PHONY: all, clean, clear, instrument

SHELL := /bin/bash
PWD := $(shell pwd)
LIB_OPENBLAS := ${PWD}/openblas/lib/
LIB_OPENBLAS_INST := ${HOME}/Documents/projects/sgxblas/attack/source/llvm/openblas_inst/lib/

CXX = clang++
CXXFLAGS += -c -std=c++17 -Wall $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv4)
CXXARGS += 

OPENCV_VERSION := 4.5.3

INSTANCE_ID ?= 0
INSTANCE_CXXFLAG += -Wall $(shell pkg-config --cflags opencv4) -lopenblas -lm -L${PWD}/instances/${INSTANCE_ID}/openblas/lib/ ${LDFLAGS}
INSTRUMENT_CXXFLAG += -Wall $(shell pkg-config --cflags opencv4) -lopenblas -lm -L${LIB_OPENBLAS_INST} ${LDFLAGS}


all: poc.out
	@echo Done!
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$<


gtsrb_resnet34: poc.out
	@echo Done!
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$< gtsrb resnet34

gtsrb_resnet50: poc.out
	@echo Done!
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$< gtsrb resnet50

gtsrb_vgg16: poc.out
	@echo Done!
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$< gtsrb vgg16

cifar10_resnet34: poc.out
	@echo Done!
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$< cifar10 resnet34

cifar10_resnet50: poc.out
	@echo Done!
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$< cifar10 resnet50

cifar10_vgg16: poc.out
	@echo Done!
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$< cifar10 vgg16

imagenet_resnet34: poc.out
	@echo Done!
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$< imagenet resnet34

imagenet_resnet50: poc.out
	@echo Done!
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$< imagenet resnet50

imagenet_vgg16: poc.out
	@echo Done!
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$< imagenet vgg16


# run a neural network on the instrumented ML codebase
# and report the branch counting results
instrument: poc_inst.out
	export LD_LIBRARY_PATH=${LIB_OPENBLAS_INST}:/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}; \
	./$<


poc_inst.out: main_inst.cpp myresnet.cpp mydataloader.cpp
	${CXX} -o $@ -DSTOP_AT=-1 -DBATCH_SIZE=1024 main_inst.cpp myresnet.cpp mydataloader.cpp ${INSTRUMENT_CXXFLAG}


clean:
	rm -f *.o

clear: clean
	rm -rf *.out
	rm -rf dump
	rm -rf opencv-{OPENCV_VERSION}
	rm -rf openblas


# For making instances
run_instance: instances/${INSTANCE_ID}/poc.out
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/instances/${INSTANCE_ID}/openblas/lib:/usr/local/lib; \
	./$< > ./instances/${INSTANCE_ID}/log.txt 2>&1

test_instance: instances/${INSTANCE_ID}/poc.out
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/instances/${INSTANCE_ID}/openblas/lib:/usr/local/lib; \
	./$<

instance: instances/${INSTANCE_ID}/poc.out
	@echo Done!
#	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$< > ./instances/${INSTANCE_ID}/log.txt 2>&1 &

instance_static: instances/${INSTANCE_ID}/poc_static.out
	@echo Done!
	./$<

instances/${INSTANCE_ID}/poc_static.out: main.cpp myresnet.cpp mydataloader.cpp instances/${INSTANCE_ID}/openblas
	${CXX} -o $@ main.cpp myresnet.cpp mydataloader.cpp \
		-Wall $(shell pkg-config --cflags opencv4) -lm \
		${PWD}/instances/${INSTANCE_ID}/openblas/lib/libopenblas.a ${LDFLAGS} \
		-DDEBUG_FLAG=false -DCONSTRUCTOR_FLAG=false -DTIMING_FLAG=false \
		-DPREDICTION_PATH=\"instances/${INSTANCE_ID}/pred_static.txt\"

instances/${INSTANCE_ID}/poc.out: main.cpp myresnet.cpp mydataloader.cpp instances/${INSTANCE_ID}/openblas
	${CXX} -o $@ main.cpp myresnet.cpp mydataloader.cpp ${INSTANCE_CXXFLAG} \
		-DDEBUG_FLAG=false -DCONSTRUCTOR_FLAG=false -DTIMING_FLAG=false \
		-DPREDICTION_PATH=\"instances/${INSTANCE_ID}/pred.txt\"

instances/${INSTANCE_ID}/openblas:
	mkdir -p ./instances/
	mkdir -p ${PWD}/instances/${INSTANCE_ID}/
	${MAKE} -C ../openblas/ 
	${MAKE} -C ../openblas/ PREFIX=${PWD}/instances/${INSTANCE_ID}/openblas install


# For general purpose
poc.out: main.o myresnet.o mydataloader.o
	${CXX} main.o myresnet.o mydataloader.o -o $@ \
	-L${LIB_OPENBLAS} ${LDFLAGS} -lopenblas -lm

main.o: main.cpp tensor.h myresnet.h mydataloader.h utils.h common.h
	${CXX} -c -Wall -DSTOP_AT=-1 -DBATCH_SIZE=4 $< -o $@
	

myresnet.o: myresnet.cpp myresnet.h tensor.h utils.h common.h
	${CXX} -c -Wall $< -o $@
	

mydataloader.o: mydataloader.cpp mydataloader.h tensor.h utils.h common.h
	${CXX} $< -o $@ ${CXXFLAGS} ${CXXARGS}
	

resnet.out: myresnet.cpp myresnet.h tensor.h utils.h
	${CXX} -Wall -o $@ $< -L${LIB_OPENBLAS} -lopenblas -lm

test_modules: resnet.out
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$<

# tensor: tensor.out
# 	./$<

# tensor.out: main.cpp tensor.h utils.h
# 	${CXX} -Wall -o $@ $< -lm

loader: loader.out
	mkdir -p ./dump
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib; \
	./$<

loader.out: loader.o
	${CXX} $< -o $@ ${LDFLAGS}

loader.o: mydataloader.cpp mydataloader.h common.h tensor.h
	${CXX} $< -o $@ ${CXXFLAGS} ${CXXARGS}


install_opencv: opencv-${OPENCV_VERSION}
	sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"; \
	sudo apt-get -y install build-essential cmake git libgtk2.0-dev pkg-config \
		libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy \
		libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
	mkdir -p opencv-${OPENCV_VERSION}/build/
	cd opencv-${OPENCV_VERSION}/build/; \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_GENERATE_PKGCONFIG=ON ..; \
	# make -j4; \
	sudo make install

# v2: use apt to install
# reference: https://linuxize.com/post/how-to-install-opencv-on-ubuntu-20-04/
install_opencv_v2:
	sudo apt update
#	python3-opencv installs opencv for system's `python3` command (not for anaconda)
	sudo apt install libopencv-dev python3-opencv

opencv-${OPENCV_VERSION}: opencv.zip
	unzip opencv.zip -d ./

opencv.zip:
	curl -L -o opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip

# test_move: test_move.out
# 	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/openblas/lib; \
# 	./$<

# test_move.out: test_move.cpp myresnet.cpp myresnet.h tensor.h common.h utils.h
# 	${CXX} -Wall -o $@ test_move.cpp myresnet.cpp -L${LIB_OPENBLAS} -lopenblas -lm


#	gcc -o fc.out -pthread fc.c ../libopenblas.a -lm
#	./fc.out

#test2: test2.c
#	gcc -o test2.out -pthread test2.c ../libopenblas.a -lm

# testcblas: test_cblas.c
# 	gcc -o testcblas.out -pthread test_cblas.c ../libopenblas.a -lm


# For evaluating ground truth

# identify an experiment configuration
GR_EXP_CONF = gtsrb
# - gtsrb: ResNet34( GTSRB )
# - imagenet_resnet: ResNet50( ImageNet )
# - imagenet_vgg: VGG-16( ImageNet )
# - fmnist: LeNet( FashionMNIST )

# GR_BR_IDX: ground-truth branch index
GR_BR_IDX = 0
GR_PATH = ${PWD}/../../attack/source/ground_truth
GR_CXX_FLAGS += -Wall $(shell pkg-config --cflags opencv4) -lopenblas -lm -L${GR_PATH}/openblas_inst/br${GR_BR_IDX}/lib/ ${LDFLAGS}

gr-instance: ${GR_PATH}/results/${GR_EXP_CONF}/br${GR_BR_IDX}/poc.out
	@echo GR instance ${GR_BR_IDX} compiled

${GR_PATH}/results/${GR_EXP_CONF}/br${GR_BR_IDX}/poc.out: main.cpp myresnet.cpp mydataloader.cpp ${GR_PATH}/openblas_inst/br${GR_BR_IDX}
	${CXX} -o $@ main.cpp myresnet.cpp mydataloader.cpp ${GR_CXX_FLAGS} \
		-std=c++17 \
		-DPOC_DIR=\"${PWD}/..\" -DSTOP_AT=-1 -DBATCH_SIZE=1024 \
		-DDEBUG_FLAG=false -DCONSTRUCTOR_FLAG=false -DTIMING_FLAG=false \
		-DPREDICTION_PATH=\"${GR_PATH}/results/${GR_EXP_CONF}/br${GR_BR_IDX}/pred.txt\"

gr-clean-instance: ${GR_PATH}/results/${GR_EXP_CONF}/clean/poc.out
	@echo GR clean instance compiled

${GR_PATH}/results/${GR_EXP_CONF}/clean/poc.out: main.cpp myresnet.cpp mydataloader.cpp
	${CXX} -o $@ main.cpp myresnet.cpp mydataloader.cpp \
	    -std=c++17 \
		-Wall $(shell pkg-config --cflags opencv4) -L${LIB_OPENBLAS} ${LDFLAGS} -lopenblas -lm \
		-DPOC_DIR=\"${PWD}/..\" -DSTOP_AT=-1 -DBATCH_SIZE=1024 \
		-DDEBUG_FLAG=false -DCONSTRUCTOR_FLAG=false -DTIMING_FLAG=false \
		-DPREDICTION_PATH=\"${GR_PATH}/results/${GR_EXP_CONF}/clean/pred.txt\"
