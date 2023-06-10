#LLVM_SRC_PATH := /usr/include/llvm-10

#LLVM_SRC_PATH2 := /usr/include/llvm-c-10

	
OBJECTS =  rtlib.o a.o


execution: 
	g++ -c rtlib.cpp -o rtlib.o  
	clang++ -fPIE -Xclang -load -Xclang ./build/DetectBranch/libDetectBranchPass.so -c b.cpp -o b.o
	g++ -c a.cpp -o a.o
	g++ a.o b.o rtlib.o -o exe


