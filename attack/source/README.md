- victim library: the ML library we are about to attack 
- target page: a page with an exploitable bit

Annotation about each project
- codehammer: a complete attack workflow # TODO
- dramlocator: consume as much irrelevant memory as possible, so that memory waylaying can hit the target page as soon as possible
- llvm: LLVM-based OpenBLAS instrumentation, getting the branching imbalance results through dynamic analysis
- ground_truth: get the ground truth effect of flipping branching instructions, through LLVM (thus compatible with the dynamic analysis results)

- capstone: (not used)

- legacy
    - bflipper: to generate a poisoned copy of the victim library with a flipped bit at the specified offset, to simulate a rowhammer attack 
    - blashammer: to search all exploitable bits from a rowhammer fuzzing output, and place the victim library at the target page, for a later rowhammer attack
        - Note: the attacker should control the aggressor rows before the target page is assigned to the victim library, so this project needs a major revision, thus moved to 'codehammer'
    - waylaying: check the current mapped physical address of a specified page in a file