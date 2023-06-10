#include <llvm/Pass.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/TypeSize.h>

using namespace llvm;

// command-line argument
// Reference: https://stackoverflow.com/questions/13626993/is-it-possible-to-add-arguments-for-user-defined-passes-in-llvm
static cl::opt<unsigned> 
    InFileBranchIndex("branch-index", cl::init(0), 
                      cl::desc("The in-file branch index for flipping"));

namespace {


struct flipBranches: public ModulePass
{
    static char ID;
    Constant *one1_const;


    flipBranches(): ModulePass(ID) { }


    bool runOnModule(Module& M) override {
        // errs() << "Hello, " << M.getModuleIdentifier() << '\n';
        // errs() << "+ in-file branch index: " << InFileBranchIndex.getValue() << '\n';

        one1_const = ConstantInt::get(Type::getInt1Ty(M.getContext()), 1);

        int br_cnt = 0;
        bool early_stop = false;

        for (Function& F: M) {
            if (F.isDeclaration()) continue;
            // errs().write_escaped(F.getName()) << '\n';

            for (BasicBlock& BB: F) {

                for (Instruction& Inst: BB) {
                    
                    // Consider conditional `br` and `select` instructions,
                    // as in pass countbranches (refer to attack/source/llvm)
                    if (auto* br_inst = dyn_cast<BranchInst>(&Inst)) {
                        if (br_inst->isConditional()) {
                            // For the `br` instruction of our interest,
                            // we insert an XOR instruction to flip the branch condition (flag),
                            // and use the flipped value as the new condition.
                            if ((br_cnt ++) == InFileBranchIndex.getValue()) {
                                Value *flag_val = br_inst->getOperand(0);
                                BinaryOperator *xor_inst = BinaryOperator::Create(
                                    BinaryOperator::Xor, flag_val, one1_const, 
                                    "flip_br", br_inst);
                                br_inst->setOperand(0, xor_inst);

                                errs() << "+ [" << InFileBranchIndex.getValue() << "] br is flipped:\t";
                                br_inst->print(errs());
                                errs() << '\n';
                                early_stop = true;
                            }
                        }
                        // unconditional branch is excluded

                    } else if (auto* sl_inst = dyn_cast<SelectInst>(&Inst)) {
                        // same logic for `select` instructions
                        if ((br_cnt ++) == InFileBranchIndex.getValue()) {
                            Value *flag_val = sl_inst->getOperand(0);
                            BinaryOperator *xor_inst;
                            if (flag_val->getType()->isIntegerTy(1)) {
                                // if select condition is not a vector, use one1_const
                                xor_inst = BinaryOperator::Create(
                                    BinaryOperator::Xor, flag_val, one1_const, 
                                    "flip_br", sl_inst);
                            } else {
                                // otherwise, create a new vector
                                ElementCount num_elems = cast<VectorType>(flag_val->getType())->getElementCount();
                                Constant *vec_one1_const = ConstantVector::getSplat(num_elems, one1_const);
                                xor_inst = BinaryOperator::Create(
                                    BinaryOperator::Xor, flag_val, vec_one1_const, 
                                    "flip_br", sl_inst);
                            }
                            sl_inst->setOperand(0, xor_inst);
                            
                            errs() << "+ [" << InFileBranchIndex.getValue() << "] select is flipped:\t";
                            sl_inst->print(errs());
                            errs() << '\n';
                            early_stop = true;
                        }
                    }

                    if (early_stop)
                        return true;
                }
            }
        }        

        errs() << "[warning] instruction of our interest is not found, bitcode not changed\n";
        return false;
    }
};

};


char flipBranches::ID = 0;
/** if a pass walks CFG without modifying it then the third argument is set to true; 
  * if a pass is an analysis pass, for example dominator tree pass, then true is 
  * supplied as the fourth argument. **/
static RegisterPass<flipBranches> X("flipbranches", "Flip Branches", false, false);

static llvm::RegisterStandardPasses Y(
    llvm::PassManagerBuilder::EP_EarlyAsPossible,
    [](const llvm::PassManagerBuilder &Builder,
       llvm::legacy::PassManagerBase &PM) { PM.add(new flipBranches()); });