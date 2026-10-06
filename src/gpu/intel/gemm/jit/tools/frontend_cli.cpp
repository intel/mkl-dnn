/*******************************************************************************
* Copyright 2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "frontend.hpp"
#include "exit_codes.hpp"
#include "internal/utils.hpp"
#include "ngen_register_allocator.hpp"

using namespace gemmstone;

template< typename T>
void writeFile(const std::string &fileName, const T &data) {
    std::ofstream ofs(fileName, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(data.data()), data.size());
}

void usage(const char *ProgramName)
{
    std::cerr << "BLAS kernel generator for Gen\n"
              << "-----------------------------\n\n"
              << "Usage:\n"
              << "  " << ProgramName << " [arguments] <level3Kern> <precisionA><precisionB><precisionC> <layoutA><layoutB><layoutC> unrollM unrollN [alpha beta] [strategy...]\n"
              << "Arguments can be any of:    -o, --out <prefix>    Write binary//asm/notes to prefix.{elf/asm/notes}\n"
              << "\n"
              << "<level3Kern> can be one of: gemm                 GEMM kernel\n"
              << "                            gemm[l|u]            SYRK/GEMMT kernel (updates only upper/lower triangle)\n"
              << "                            gemm[l|u]h           HERK/GEMMT kernel (updates upper/lower triangle, forces real diagonal)\n"
              << "                            batchs gemm          Batched GEMM kernel (strided) \n"
              << "                            batchn gemm          Batched GEMM kernel (non-strided) \n"
              << "                            batchnv gemm         Batched GEMM kernel (non-strided, variable) \n"
              << "                            offset[ab][c] gemm   GEMM kernel with A/B/C offsets\n"
              << "                            trmm[l|r][l|u]       TRMM kernel\n";
    std::cerr << "                            hpl gemm             HPL-type GEMM kernel\n"
              << "  <copyKern> can be one of: copy                 GEMM copy/scaling kernel\n"
              << "                            copyc                GEMM copy/scaling kernel, with conjugation\n"
              << "                            copy[l|u]s           SYMM copy/scaling kernel\n"
              << "                            copy[l|u]h           HEMM copy/scaling kernel\n"
              << "                            copy[l|u]t[n|u]      TRMM copy/scaling kernel (n/u = nonunit/unit)\n";
    std::cerr << "        Precisions include: S/D/C/Z              Single/double/complex single/complex double\n"
              << "                            H                    Half precision (IEEE fp16)\n"
              << "                            B                    bfloat16\n"
              << "                            Q                    fp8 E5M2 (\"bf8\")\n"
              << "                            q                    fp8 E4M3 (\"hf8\")\n"
              << "                            T                    tf32\n"
              << "                            J                    Complex half precision\n"
              << "                            K                    Complex bfloat16\n"
              << "                            U                    Complex tf32\n"
              << "                            F/f                  Signed/unsigned 4-bit integer\n"
              << "                            O/o                  Signed/unsigned 8-bit integer\n"
              << "                            W/w                  Signed/unsigned 16-bit integer\n"
              << "                            I/i                  Signed/unsigned 32-bit integer\n"
              << "           Layouts include: N                    Column major\n"
              << "                            T                    Row major\n"
              << "                            A                    Packed A matrix (packed row panels of height unrollN)\n"
              << "                            B                    Packed B matrix (packed column panels of height unrollM)\n"
              << "* alpha, beta can each be either a number (if fixed) or - (variable).\n"
              << "      If not specified, the default is alpha = 1, beta = 0 (1 for global k-parallel kernels).\n";
    std::cerr << "\nOptions:\n"
              << "  -o, --out-file <file>        Write result.info to <file>\n";
    exit(1);
}

struct FrontendArgs {
    const char *programName = nullptr;
    const char *out = nullptr;
};


int parseArgs(FrontendArgs &fArgs, int argc, char **argv) {
    int argIdx = 0;
    fArgs.programName = argv[argIdx++];

    struct Handler {
        int required = 0;
        std::function<void()> parse;
    };

    const std::unordered_map<std::string, Handler> argHandlers = {
        {"--out-file", {1, [&]() { fArgs.out = argv[argIdx++]; }}},
        {"-o",         {1, [&]() { fArgs.out = argv[argIdx++]; }}},
    };

    while (argIdx < argc) {
        auto it = argHandlers.find(argv[argIdx]);
        if (it == argHandlers.end()) break;
        argIdx++;
        auto &handler = it->second;
        if (handler.required + argIdx > argc) {
          std::cerr << "Error: option " << it->first << " requires input.\n";
          exit(1);
        }
        handler.parse();
    }
    return argIdx;
}

int main(int argc, char **argv)
{

    FrontendArgs fArgs;
    int parse_idx = parseArgs(fArgs, argc, argv);
    argc -= parse_idx;
    argv += parse_idx;
    if (argc <= 0) usage(fArgs.programName);

    std::exception_ptr eptr;
    int status = 0;

    // Convert argv into a space-separated string
    std::string kernelArgs;
    for (int i = 0; i < argc; i++) {
        kernelArgs += argv[i];
        if (i < argc-1) kernelArgs += " ";
    }
    try {
        std::unique_ptr<frontend::Generator> generator = frontend::Generator::Create();
        auto result = generator->generate(kernelArgs);
        if (!result.success) {
            std::cerr << "Generation error: " << result.info << std::endl;
            return EXIT_CODE_MISC_ERROR;
        }
        if (fArgs.out) {
            std::string infoFile = std::string(fArgs.out) + ".note";
            std::string elfFile = std::string(fArgs.out) + ".elf";
            std::string asmFile = std::string(fArgs.out) + ".asm";
            writeFile(infoFile, result.info + "\n");
            if(!result.asmStr.empty()) writeFile(asmFile, result.asmStr);
            if(!result.binary.empty()) writeFile(elfFile, result.binary);
        } else {
            if (!result.asmStr.empty()) {
                std::cout << result.info << std::endl;
                std::cout << result.asmStr;
            } else {
                std::cerr << result.info << std::endl;
                std::cout.write((char *) result.binary.data(), result.binary.size());
            }
            std::cout.flush();
        }
    } catch (const ngen::out_of_registers_exception &e) {
        eptr = std::current_exception();
        status = EXIT_CODE_OUT_OF_REGISTERS;
    } catch (const stub_exception &e) {
        eptr = std::current_exception();
        status = EXIT_CODE_STUB;
    } catch (const hw_unsupported_exception &e) {
        eptr = std::current_exception();
        status = EXIT_CODE_HW_UNSUPPORTED;
    } catch (...) {
        eptr = std::current_exception();
        status = EXIT_CODE_MISC_ERROR;
    }

    if (eptr) {
        try {
            std::rethrow_exception(eptr);
        } catch (const std::exception &e) {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
        }
    }
    return status;
}
