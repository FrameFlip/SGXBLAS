sgxblas/attack/source/llvm: dynamic analysis on machine learning codebase
- **Step 1** | compile the openblas library
  ```bash
  make init
  ```
- **Step 2** | generate LLVM IR for target source files
- apply instrumentation on IR files through LLVM Pass
- re-compile using the instrumented IR, to get an instrumented shared library
  ```bash
  make inst_openblas
  ```
- **Step 3** | use the modified library to run a neural network
- get the dynamic analysis results
  ```bash
  make -C ../../../poc/infras/ instrument
  ```