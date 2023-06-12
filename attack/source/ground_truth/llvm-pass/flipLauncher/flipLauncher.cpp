#include <llvm/Pass.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include <cstdlib> // system()
// for reading and saving branch info
#include <fstream>
#include <string>
#include <llvm/Support/FileSystem.h>
// for constructing the command
#include <sstream>

#ifndef BRANCH_INFO_FILE
#define BRANCH_INFO_FILE "/home/xxx/SGXBLAS/attack/source/ground_truth/br_info.tmp"
#endif

using namespace llvm;


namespace {


struct flipLauncher: public ModulePass
{
    static char ID;
    int br_idx;         // branch instruction index


    flipLauncher(): ModulePass(ID), br_idx(0) {
        // initialize `br_idx`
        std::ifstream info_file(BRANCH_INFO_FILE);

        if (!info_file.good()) {
            outs() << "branch info file not found, set index to 0\n";
            br_idx = 0;
        } else {
            // branch info file format: (tab used as delimiter)
            // br_idx   module:function:basicblock  infile_idx
            //   instruction
            int infile_index;
            std::string branch_info;
            std::string inst;
            while (info_file >> br_idx >> branch_info >> infile_index) {
                info_file.ignore(3);    // ignore the newline and two spaces
                std::getline(info_file, inst);
                // if (infile_index == 0)
                //     outs() << ">> " << branch_info << '\n';
                // outs() << br_idx << '\t' << inst << '\n';
            }
            // need to increment br_idx of the last instruction by 1
            outs() << "branch index is set to " << (++ br_idx) << '\n'; 
        }

        info_file.close();
    }


    bool runOnModule(Module& M) override {
        std::string module_name = M.getName().str();
        // extract the prefix (exclude the file extension e.g. .bc .ll)
        std::string module_prefix;
        std::size_t dot_pos = module_name.find('.');
        if (dot_pos == std::string::npos)
            module_prefix = module_name;
        else    
            module_prefix = module_name.substr(0, dot_pos);

        outs() << "Hello, " << module_prefix << " / " << module_name << '\n';

        int infile_idx = 0;
        std::error_code EC;
        raw_fd_ostream info_file_out(BRANCH_INFO_FILE, EC, sys::fs::OF_Append);
        std::string command_prefix = "opt -enable-new-pm=0 -load ${HOME}/xxxx/SGXBLAS/attack/source/ground_truth/llvm-pass/build/flipBranches/libflipBranches.so -flipbranches -o ";

        for (Function& F: M) {
            if (F.isDeclaration()) continue;

            for (BasicBlock& BB: F) {

                for (Instruction& Inst: BB) {

                    // For each conditional branch and select instruction,
                    // we flip the instruction, and use the contaminated bitcode to compile a 
                    // separate openblas library
                    if (auto *br_inst = dyn_cast<BranchInst>(&Inst)) {
                        if (br_inst->isConditional()) {
                            info_file_out << br_idx << '\t' 
                                << M.getName().str() << ':' << F.getName().str() 
                                << ':' << BB.getName().str() << '\t'
                                << (infile_idx) << '\n';
                            br_inst->print(info_file_out);
                            info_file_out << '\n';

                            // command 1: apply flipbranches pass to the module,
                            // setting --branch-index to `infile_idx`
                            std::stringstream command_ss;
                            command_ss << command_prefix
                            // output name
                            << module_prefix << "_inst." << br_idx << ".bc"
                            // input name
                            << " " << module_name
                            // command-line argument
                            << " --branch-index=" << infile_idx;

                            // outs() << command_ss.str() << '\n';
                            int status_code = system(command_ss.str().c_str());

                            ++ br_idx;
                            ++ infile_idx;
                        }
                        // unconditional branch is excluded
                    } else if (auto *sl_inst = dyn_cast<SelectInst>(&Inst)) {
                        info_file_out << br_idx << '\t' 
                            << M.getName().str() << ':' << F.getName().str() 
                            << ':' << BB.getName().str() << '\t'
                            << (infile_idx) << '\n';
                        sl_inst->print(info_file_out);
                        info_file_out << '\n';

                        std::stringstream command_ss;
                        command_ss << command_prefix
                        // output name
                        << module_prefix << "_inst." << br_idx << ".bc"
                        // input name
                        << " " << module_name
                        // command-line argument
                        << " --branch-index=" << infile_idx;

                        // outs() << command_ss.str() << '\n';
                        int status_code = system(command_ss.str().c_str());

                        ++ br_idx;
                        ++ infile_idx;
                    }
                }
            }
        }
        

        info_file_out.close();

        return false;
    }
};

};


char flipLauncher::ID = 0;
/** if a pass walks CFG without modifying it then the third argument is set to true; 
  * if a pass is an analysis pass, for example dominator tree pass, then true is 
  * supplied as the fourth argument. **/
static RegisterPass<flipLauncher> X("fliplauncher", "Flip Branch Launcher", false, false);

static llvm::RegisterStandardPasses Y(
    llvm::PassManagerBuilder::EP_EarlyAsPossible,
    [](const llvm::PassManagerBuilder &Builder,
       llvm::legacy::PassManagerBase &PM) { PM.add(new flipLauncher()); });