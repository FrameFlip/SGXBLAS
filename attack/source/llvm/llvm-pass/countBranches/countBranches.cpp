#include <llvm/Pass.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/TypeSize.h>

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

// for handling br_idx.tmp
#include <sys/stat.h>
#include <string>
#include <fstream>

// Use absolute path for br_idx file
#ifndef BRANCH_INDEX_FILE
#define BRANCH_INDEX_FILE "/home/shawn233/Documents/projects/sgxblas/attack/source/llvm/br_idx.tmp"
#endif

using namespace llvm;


namespace {


bool check_file_existence(const char *path) {
    struct stat buffer;
    return ( stat(path, &buffer) == 0 );
}


struct countBranches: public ModulePass
{
    static char ID;
    int br_idx;             // index of a branching instruction
    int n_ignored_insts;    // total amount of ignored instructions

    Constant *zero64_const;
    Constant *one32_const;
    Constant *two32_const;
    ArrayType* global_arr_t;
    GlobalVariable *branch_counter;     // count the total number of branching
    GlobalVariable *taken_counter;      // count the number of times taking the if-branch


    countBranches(): ModulePass(ID), br_idx(0), n_ignored_insts(0),
        zero64_const(nullptr), one32_const(nullptr), global_arr_t(nullptr),
        branch_counter(nullptr), taken_counter(nullptr) {
        // initialize `br_idx`
        std::ifstream idx_file(BRANCH_INDEX_FILE);
        int n_ignored;
        if (! idx_file.good() ) {
            outs() << "branch index file not found, set index to 0\n";
            br_idx = 0;
        } else {
            // `br_idx` takes the value from the last instrumented object file
            std::string obj_filename;
            outs() << "instrumented\tn_branches\tn_ignored\n";
            // Reference: https://stackoverflow.com/questions/43956124/c-while-loop-to-read-from-input-file
            while (idx_file >> obj_filename >> br_idx >> n_ignored) {
                outs() << obj_filename << '\t' << br_idx << '\t' << n_ignored << '\n';
            }
            outs() << "branch index is set to " << br_idx << '\n';
        }
        idx_file.close();
    }


    /**
     * @brief instrumentation to monitor runtime branching status (only effect
     * on conditional branches)
     * 
     * @param context module context
     * @param branch_inst ptr to an instance of either BranchInst or SelectInst
     */
    void insertBeforeBranch(LLVMContext& context,
        Instruction* branch_inst) 
    {
        if ( isa<BranchInst>(branch_inst) || 
             (isa<SelectInst>(branch_inst) && 
              cast<SelectInst>(branch_inst)->getOperand(0)->getType()->isIntegerTy(1) )) {
            // simple case: branchinst (only supports i1 condition) or selectinst with i1 condition,
            //              for selectinst with vectorized condition, direct to a more complex case

            IRBuilder<> builder(branch_inst);

            // Increment branch_counter[br_idx] by 1
            // Reference: https://stackoverflow.com/questions/26787341/inserting-getelementpointer-instruction-in-llvm-ir
            //            https://llvm.org/doxygen/classllvm_1_1ConstantInt.html
            //            https://llvm.org/docs/GetElementPtr.html#why-is-the-extra-0-index-required
            Constant *br_idx_const = ConstantInt::get(Type::getInt64Ty(context), br_idx);
            
            Value *gep1 = builder.CreateInBoundsGEP(
                global_arr_t, 
                branch_counter, 
                {zero64_const, br_idx_const}); // first index is 0 because global variable is a pointer to array
            LoadInst *load_val1 = builder.CreateAlignedLoad(
                Type::getInt32Ty(context), gep1, MaybeAlign(4));

            Value *inc_val1 = builder.CreateNSWAdd(load_val1, one32_const, formatv("inc_br{0}counter", br_idx));
            
            Value *gep2 = builder.CreateInBoundsGEP(
                global_arr_t, 
                branch_counter, 
                {zero64_const, br_idx_const}); // first index is 0 because global variable is a pointer to array
            StoreInst *store_inc1 = builder.CreateStore(inc_val1, gep2);

            // Similary, change taken_counter[br_idx] in regard of the branching condition
            // if the if-branch is taken (condition = true), then increment by 1
            // otherwise, make no change
            Value *gep3 = builder.CreateInBoundsGEP(
                global_arr_t, taken_counter, {zero64_const, br_idx_const});
            LoadInst *load_val2 = builder.CreateAlignedLoad(
                Type::getInt32Ty(context), gep3, MaybeAlign(4));

            Value *cond_flag = branch_inst->getOperand(0);
            Value *cond_flag_ext = builder.CreateZExt(cond_flag, Type::getInt32Ty(context), "cond");

            Value *inc_val2 = builder.CreateNSWAdd(load_val2, cond_flag_ext, formatv("inc_br{0}taken", br_idx));
            
            Value *gep4 = builder.CreateInBoundsGEP(
                global_arr_t, 
                taken_counter,
                {zero64_const, br_idx_const}); // first index is 0 because global variable is a pointer to array
            StoreInst *store_inc2 = builder.CreateStore(inc_val2, gep4);
        
        } else if (isa<SelectInst>(branch_inst)) {
            // complex case: selectinst with vectorized condition
            // for a vectorized condition, the counter treats each element individually
            SelectInst *sl_inst = cast<SelectInst>(branch_inst);
            Value *cond_val = sl_inst->getOperand(0);

            VectorType *cond_vec_ty;
            if (cond_vec_ty = dyn_cast<VectorType>(cond_val->getType()))
                ;
            else {
                outs() << "[warning] Instruction ignored: The condition is not of VectorType. Type: ";
                cond_val->getType()->print(outs());
                outs() << '\n';

                ++ n_ignored_insts;

                // Note that `br_idx` is not increased in this case
                return;
            }

            // In regard of the difficulty in `taken_counter` evaluation,
            // we ignore vectors of more than two elements and report the amount.
            
            unsigned int num_elems = cond_vec_ty->getElementCount().getValue(0);
            if (num_elems != 2) {
                outs() << "[warning] Instruction ignored: The condition is a vector of > 2 elements. Condition vector size: " << num_elems << '\n';

                ++ n_ignored_insts;

                // Note that `br_idx` is not increased in this case
                return;
            }

            IRBuilder<> builder(branch_inst);

            // Increment branch_counter[br_idx] by num_elems of the condition vector
            Constant *br_idx_const = ConstantInt::get(Type::getInt64Ty(context), br_idx);
            
            Value *gep1 = builder.CreateInBoundsGEP(
                global_arr_t, 
                branch_counter, 
                {zero64_const, br_idx_const}); // first index is 0 because global variable is a pointer to array
            LoadInst *load_val1 = builder.CreateAlignedLoad(
                Type::getInt32Ty(context), gep1, MaybeAlign(4));

            // increment branch_counter[br_idx] by exactly two (since we ignore other cases)
            Value *inc_val1 = builder.CreateNSWAdd(load_val1, two32_const, formatv("inc_br{0}counter", br_idx));
            
            Value *gep2 = builder.CreateInBoundsGEP(
                global_arr_t, 
                branch_counter, 
                {zero64_const, br_idx_const}); // first index is 0 because global variable is a pointer to array
            StoreInst *store_inc1 = builder.CreateStore(inc_val1, gep2);

            // Similary, change taken_counter[br_idx] in regard of the branching condition
            // if the if-branch is taken (condition = true), then increment by 1;
            // otherwise, make no change.
            // For vectorized condition, if the size is two, we sum up the two conditions.
            // For a vector of more than two elements, we ignore it and report the amount.
            Value *gep3 = builder.CreateInBoundsGEP(
                global_arr_t, taken_counter, {zero64_const, br_idx_const});
            LoadInst *load_val2 = builder.CreateAlignedLoad(
                Type::getInt32Ty(context), gep3, MaybeAlign(4));

            // // here the index must use UL as the postfix, to avoid parameter list ambiguity
            Value *cond_val0 = builder.CreateExtractElement(cond_val, 0UL, formatv("cond{0}_0", br_idx));
            Value *cond_val1 = builder.CreateExtractElement(cond_val, 1UL, formatv("cond{0}_1", br_idx));
            Value *cond_val0_ext = builder.CreateZExt(cond_val0, Type::getInt32Ty(context), formatv("cond{0}_0ext", br_idx));
            Value *cond_val1_ext = builder.CreateZExt(cond_val1, Type::getInt32Ty(context), formatv("cond{0}_1ext", br_idx));
            Value *cond_val_sum = builder.CreateNSWAdd(cond_val0_ext, cond_val1_ext, formatv("cond{0}_sum", br_idx));

            // Value *cond_flag_ext = builder.CreateZExt(cond_flag, Type::getInt32Ty(context), "cond");

            Value *inc_val2 = builder.CreateNSWAdd(load_val2, cond_val_sum, formatv("inc_br{0}taken", br_idx));
            
            Value *gep4 = builder.CreateInBoundsGEP(
                global_arr_t, 
                taken_counter,
                {zero64_const, br_idx_const}); // first index is 0 because global variable is a pointer to array
            StoreInst *store_inc2 = builder.CreateStore(inc_val2, gep4);
        }

        ++ br_idx;
    }


    bool runOnModule(Module& M) override {
        outs() << "Hello, " << M.getModuleIdentifier() << '\n';

        // initialize
        zero64_const = ConstantInt::get(Type::getInt64Ty(M.getContext()), 0);
        one32_const = ConstantInt::get(Type::getInt32Ty(M.getContext()), 1);
        two32_const = ConstantInt::get(Type::getInt32Ty(M.getContext()), 2);

        // define a new global array [100 x i32]: branch_counter
        // effect to llvm ir:
        // [+] @branch_counter = dso_local global [100 x i32] zeroinitializer, align 16
        const int arr_size = 500; // assume at most 100 branches
        LLVMContext& context = M.getContext();
        global_arr_t = ArrayType::get(Type::getInt32Ty(context), arr_size);
        
        branch_counter = new GlobalVariable(
            /*Module=*/M, 
            /*Type=*/global_arr_t, 
            /*isConstant=*/false, 
            /*Linkage=*/GlobalValue::ExternalLinkage, 
            // zeroinitializer reference: https://stackoverflow.com/questions/23330018/llvm-global-integer-array-zeroinitializer
            // when br_idx != 0, set initializer to 0, to make sure it is only defined once
            /*Initializer=*/(br_idx == 0)? ConstantAggregateZero::get(global_arr_t): 0,
            /*Name=*/"branch_counter");
            // /*InsertBefore=*/nullptr,
            // /*TLMode=*/GlobalValue::NotThreadLocal,
            // /*AddressSpace=*/0U, // Docs suggest None here, but None won't compile
            // /*isExternallyInitialized=*/(br_idx == 0)? false: true);
        branch_counter->setAlignment(MaybeAlign(16));
        // shawn233: dso_local will cause link-stage issues: 
        // > relocation R_X86_64_PC32 against symbol `branch_counter' can not be used when 
        // > making a shared object; recompile with -fPIC
        // So do not make it DSOLocal.
        // Reference: https://stackoverflow.com/questions/70405403/how-to-create-global-variable-as-dso-local-in-llvm-ir-builder-for-c
        // branch_counter->setDSOLocal(true);

        // Similarly, we define global variable taken_counter
        taken_counter = new GlobalVariable(
            M, global_arr_t, false, GlobalValue::ExternalLinkage, 
            (br_idx == 0)? ConstantAggregateZero::get(global_arr_t): 0, 
            "taken_counter"); 
            // nullptr, GlobalValue::NotThreadLocal, 0U, 
            // (br_idx == 0)? false: true);
        taken_counter->setAlignment(MaybeAlign(16));

        // view the global variable list
        const Module::GlobalListType& global_list = M.getGlobalList();
        outs() << "< Global Variable List >\n";
        for (auto& global_var: global_list) {
            outs() << global_var.getName() << '\t';
            global_var.getType()->print(outs());
            outs() << '\n';
        }

        // analyze conditional branches
        outs() << "< Conditional Branches >\n";

        for (Function& F: M) {
            if (F.isDeclaration()) continue;
            outs().write_escaped(F.getName()) << '\n';
            
            for (BasicBlock& BB: F) {
                
                for (Instruction& Inst: BB) {

                    if (auto* br_inst = dyn_cast<BranchInst>(&Inst)) {
                        if (br_inst->isConditional()) {
                            // find the immediate previous icmp instruction
                            const Instruction* tmp_inst = br_inst;
                            while ( (tmp_inst = tmp_inst->getPrevNonDebugInstruction()) ) {
                                if (auto* cmp_inst = dyn_cast<CmpInst>(tmp_inst)) {
                                    // icmp hit
                                    outs() << br_idx << ")\n";
                                    cmp_inst->print(outs());
                                    outs() << '\n';
                                    br_inst->print(outs());
                                    outs() << "\n\n";
                                    break;
                                }
                            }

                            insertBeforeBranch(M.getContext(), br_inst);
                        }
                        // unconditional branch is excluded
                    }
                    // select instructions are common in llvm IR, translated from
                    // conditional value assignment
                    else if (auto* sl_inst = dyn_cast<SelectInst>(&Inst)) {
                        // find the immediate previous icmp instruction
                        const Instruction* tmp_inst = sl_inst;
                        while ( (tmp_inst = tmp_inst->getPrevNonDebugInstruction()) ) {
                            if (auto* cmp_inst = dyn_cast<CmpInst>(tmp_inst)) {
                                // icmp hit
                                outs() << br_idx << ")\n";
                                cmp_inst->print(outs());
                                outs() << '\n';
                                sl_inst->print(outs());
                                outs() << "\n\n";
                                break;
                            }
                        }

                        insertBeforeBranch(M.getContext(), sl_inst);
                    }
                }
            }
        }
        
        outs() << "total cond branches: " << br_idx << '\n';
        outs() << "total ignored instructions: " << n_ignored_insts << '\n';

        // write br_idx to branch index file
        // other object files can continue on the index
        std::ofstream idx_file(BRANCH_INDEX_FILE, std::ios::app);
        idx_file << M.getModuleIdentifier() << '\t' << br_idx << '\t' << n_ignored_insts << '\n';
        idx_file.close();

        return true;
    }
};

};


char countBranches::ID = 0;
/** if a pass walks CFG without modifying it then the third argument is set to true; 
  * if a pass is an analysis pass, for example dominator tree pass, then true is 
  * supplied as the fourth argument. **/
static RegisterPass<countBranches> X("countbranches", "Count Branches", false, false);

static llvm::RegisterStandardPasses Y(
    llvm::PassManagerBuilder::EP_EarlyAsPossible,
    [](const llvm::PassManagerBuilder &Builder,
       llvm::legacy::PassManagerBase &PM) { PM.add(new countBranches()); });