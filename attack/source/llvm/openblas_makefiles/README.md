- init: openblas first-time compile, set 
    - CC = clang
    - ONLY_CBLAS = 1
    - NO_LAPACK = 1
    - BUILD_LAPACK_DEPRECATED = 0
  in Makefile.rule
- fast: after first-time compile, in regard to compile speed,
    comment out tests, $(RELA), and netlib in target 'all' in Makefile