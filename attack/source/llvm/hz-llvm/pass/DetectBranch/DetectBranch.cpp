#include "llvm/Pass.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"

#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <vector>

using namespace llvm;

namespace {
    struct DetectBranchPass : public ModulePass {
    public:
        IntegerType *Int32Ty;


        static char ID;

        DetectBranchPass() : ModulePass(ID) {}

        bool runOnModule(Module &M) override;

    };
}

char DetectBranchPass::ID = 0;

//extern void getCondVal();

//extern std::map<BranchInst, int>* mapPointer;
/*
extern std::string getBranchCondString(Instruction*);
*/
/*
std::string getBranchString(Instruction *TI) {
   BranchInst *BI = dyn_cast<BranchInst>(TI);
   if (!BI || !BI->isConditional())
     return std::string();
  
   Value *Cond = BI->getCondition();
   ICmpInst *CI = dyn_cast<ICmpInst>(Cond);
   if (!CI)
     return std::string();
  
   std::string result;
   raw_string_ostream OS(result);
   OS<< "br " ;
   CI->getOperand(0)->getType()->print(OS,true);
   OS << CmpInst::getPredicateName(CI->getPredicate()) << "_";
   CI->getOperand(0)->getType()->print(OS, true);
  
   Value *RHS = CI->getOperand(1);
   ConstantInt *CV = dyn_cast<ConstantInt>(RHS);
   if (CV) {
     if (CV->isZero())
       OS << "_Zero";
     else if (CV->isOne())
       OS << "_One";
     else if (CV->isMinusOne())
       OS << "_MinusOne";
     else
       OS << "_Const";
   }
   OS<<(*TI);
   OS.flush();
   return result;
 }
 */
 
static int BrCnt = 0;

bool DetectBranchPass::runOnModule(Module &M) {

    LLVMContext &Ctx = M.getContext();
    Int32Ty = IntegerType::getInt32Ty(Ctx);

/*
    auto getCondValFunc = M.getOrInsertFunction(
        "getCondVal",Type::getVoidTy(Ctx),Type::getInt32Ty(Ctx)
    );
    
    */
    
auto getCondValFunc = M.getOrInsertFunction(
        "_Z10getCondValiPKc",Type::getVoidTy(Ctx),Type::getInt32Ty(Ctx), Type::getInt8PtrTy(Ctx)
    );
    

    for (auto &F: M)
        for (auto &BB: F) {
            for (auto &I: BB) {
                if (auto* Br=dyn_cast<BranchInst>(&I)) {
                    if (Br->isConditional()) {
//			errs()<<(*Br)<<"\n";
                        Value *Cond = Br->getCondition();
                        if (Cond && Cond->getType()->isIntegerTy() && !isa<ConstantInt>(Cond)) {
                            if (CmpInst *CondInst = dyn_cast<CmpInst>(Cond)) {
                                IRBuilder<> IRB(CondInst);
                                IRB.SetInsertPoint(&BB, ++IRB.GetInsertPoint());
                                
				 std::string Brstr;
				 llvm::raw_string_ostream tmpOS(Brstr);
				 tmpOS << (*Br);
				 tmpOS.flush();


                                
                                Value* args1 = {CondInst};
//                                Value* args2;

				 Value *args2 = IRB.CreateGlobalStringPtr(Brstr);
//Using a Vector instead of ArrayRef
// const std::vector<llvm::Value *> args{strPointer};
// builder.CreateCall(instrumentation_function, args);
//                                Value * args2 = IRB.CreateGlobalString(StringRef(Brstr),"BranchName"+(++BrCnt));
//                                Value* args2 = {StringRef(Brstr)};
//                                ArrayRef<Value*> args;

                           errs()<<(*Br)<<"|"<<Brstr<<"|"<<BrCnt<<"\n";

 				 std::vector<Value*> args; 
				 args.push_back(args1);
				 args.push_back(args2);
                                
                                IRB.CreateCall(getCondValFunc,args);
                            }

                        }

                    }
                }
            }
        }

    return true;
}

static void registerDetectBranchPass(const PassManagerBuilder &,
                                     legacy::PassManagerBase &PM) {
    PM.add(new DetectBranchPass());
}

////Q1 2 passes?

static RegisterStandardPasses
        RegisterDetectBranchPass(PassManagerBuilder::EP_OptimizerLast,
                                 registerDetectBranchPass);

static RegisterStandardPasses
        RegisterDetectBranchPass0(PassManagerBuilder::EP_EnabledOnOptLevel0,
                                  registerDetectBranchPass);
