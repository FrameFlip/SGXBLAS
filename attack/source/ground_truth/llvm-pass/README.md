Compile:

```bash
mkdir build && cd build
cmake ../
make
```

The compiled LLVM Pass will be at ground_truth/build/flipLauncher/flipLauncher.so
(flipBranches is automatically invoked by flipLauncher.)

Usage: (for the newest stable LLVM version: llvm-14)

> WARNING: for lower LLVM versions (e.g. llvm-10), remove the -enable-new-pm=0 option.

```bash
opt -enable-new-pm=0 -load /path/to/libflipLauncher.so -fliplauncher -o <output bitcode> <input IR file>
```

IR file can either be a bitcode file (\*.bc) or a human readable IR file (\*.ll). 
Note that the generated file will always be a bitcode file. 