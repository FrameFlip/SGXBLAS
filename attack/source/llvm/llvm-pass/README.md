Compile:

```bash
mkdir build && cd build
cmake ../
make
```

The compiled LLVM Pass will be at llvm-pass/build/countBranches/libcountBranches.so

Usage:

```bash
opt -load /path/to/libcountBranches.so -countbranches -o <output bitcode> <input IR file>
```

IR file can either be a bitcode file (\*.bc) or a human readable IR file (\*.ll). 
Note that the generated file will always be a bitcode file. 