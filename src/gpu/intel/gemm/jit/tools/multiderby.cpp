/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>

#include "derby_utils.hpp"
#include "multiopt/agent.hpp"
#include "multiopt/data.hpp"
#include "multiderby_client.hpp"
#include "multiderby_common.hpp"
#include "multiderby_optimizer.hpp"
#include "utils.hpp" 
#include "vt100_codes.hpp"

#ifdef _WIN32
#include <conio.h>
#include <io.h>
#else
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

using namespace std::string_view_literals;


std::vector<std::byte> readFile(const std::filesystem::path &path)
{
    std::ifstream fs(path, std::ios::binary | std::ios::ate);
    if (!fs) fail("Could not read file " + path.string());

    auto bytes = fs.tellg();
    fs.seekg(0);
    std::vector<std::byte> data(bytes);
    fs.read((char *) data.data(), bytes);

    return data;
}

void writeFile(const std::filesystem::path &path, const std::vector<std::byte> &data)
{
    /* Write to a temporary file and rename into place so a checkpoint is never left truncated
       (e.g. if the process is killed mid-write), which would otherwise make it undeserializable. */
    auto tmpPath = path;
    tmpPath += ".tmp";

    {
        std::ofstream fs(tmpPath, std::ios::binary);
        if (!fs) fail("Could not write file " + tmpPath.string());
        fs.write((const char *) data.data(), data.size());
        if (!fs) fail("Could not write file " + tmpPath.string());
    }

    std::error_code ec;
    std::filesystem::rename(tmpPath, path, ec);
    if (ec) fail("Could not write file " + path.string());
}

void writeFile(const std::filesystem::path &path, const std::string &data)
{
    std::ofstream fs(path);
    if (!fs) fail("Could not write file " + path.string());
    fs << data;
}

void writeFile(const std::filesystem::path &path, const std::vector<std::string> &data)
{
    std::ofstream fs(path);
    if (!fs) fail("Could not write file " + path.string());
    for (auto &line: data)
        fs << line << std::endl;
}

void terminalTeardown();

// flag to make terminalTeardown idempotent
bool terminalActive = false;

#ifdef _WIN32
DWORD OriginalInputMode;

bool terminalSetup()
{
    if (!_isatty(_fileno(stdin))) return false;

    if (auto hStdin = GetStdHandle(STD_INPUT_HANDLE); hStdin != INVALID_HANDLE_VALUE)
        if (GetConsoleMode(hStdin, &OriginalInputMode) && SetConsoleMode(hStdin, ENABLE_PROCESSED_INPUT)) {
            std::cerr << VT_ALT_SCREEN_ON << std::flush;
            terminalActive = true;
            atexit(terminalTeardown);
        }

    return true;
}

void terminalTeardown()
{
    if (!terminalActive) return;
    terminalActive = false;
    std::cerr << VT_ALT_SCREEN_OFF << std::flush;
    SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), OriginalInputMode);
}

char terminalInput(bool block = false)
{
    return (block || _kbhit()) ? _getch() : '\0';
}
#else
termios OriginalTIOS;
bool terminalSetup()
{
    if (!isatty(STDIN_FILENO)) return false;

    tcgetattr(STDIN_FILENO, &OriginalTIOS);

    auto newTIOS = OriginalTIOS;
    newTIOS.c_lflag &= (~ICANON & ~ECHO);

    tcsetattr(STDIN_FILENO, TCSANOW, &newTIOS);
    fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);

    std::cerr << VT_ALT_SCREEN_ON << std::flush;
    terminalActive = true;
    atexit(terminalTeardown);

    return true;
}

void terminalTeardown()
{
    if (!terminalActive) return;
    terminalActive = false;
    std::cerr << VT_ALT_SCREEN_OFF << std::flush;
    tcsetattr(STDIN_FILENO, TCSANOW, &OriginalTIOS);
}

char terminalInput(bool block = false)
{
    char in = 0;
    while (read(STDIN_FILENO, &in, 1) != 1) {
        in = 0;
        if (block)
            usleep(10000);
        else
            break;
    }
    return in;
}
#endif

int parseTime(const char *s)
{
    char *end;
    int seconds = std::strtol(s, &end, 10);
    switch (*end) {
        case 'd': seconds *= 24;
        case 'h': seconds *= 60;
        case 'm': seconds *= 60;
        case 's':
        case 0: break;
        default: throw std::runtime_error("Unknown time format");
    }
    return seconds;
};

std::string formatTime(int seconds)
{
    std::stringstream s;
    double time = seconds;
    char units = 's';
    if (time >= 60) {
        time /= 60, units = 'm';
        if (time >= 60) {
            time /= 60, units = 'h';
            if (time >= 24)
                time /= 24, units = 'd';
        }
    }
    s << std::fixed << std::setprecision(1) << time << units;
    return s.str();
}

std::string defaultFileBasename(const std::string &stem)
{
    std::string base = "md_";

    switch (currentHW()) {
#define CASE(x) case ngen::HW::x: base += #x; break;
        CASE(Gen12LP)
        CASE(XeHP)
        CASE(XeHPG)
        CASE(XeHPC)
        CASE(Xe2)
        CASE(Xe3)
        CASE(Xe3p)
#undef CASE
        default: gemmstone::stub(); break;
    }
    base += '_';
    base += stem;
    for (auto &c: base) if (c == ' ') c = '_';
    return base;
}

std::filesystem::path defaultCheckpointPath(const std::string &stem)
{
    std::filesystem::path path;

#ifndef _WIN32
    if (auto e = getenv("XDG_CACHE_HOME"))
        path = e;
    else {
        path = getenv("HOME");
        path /= ".cache";
    }
#endif

    path /= defaultFileBasename(stem) + ".checkpoint";
    return path;
}

std::filesystem::path defaultOutputPath(const std::string &stem)
{
    for (int n = 0; n < 1024; n++) {
        std::filesystem::path path = defaultFileBasename(stem) + '.' + std::to_string(n) + ".data";
        if (std::error_code ec; std::filesystem::exists(path, ec))
            continue;
        return path;
    }
    return {};
}

[[noreturn]] void usage(const char *pname)
{
    std::cerr << "Multiderby: flexible autotuner for GEMM-like kernels.\n"
                 "\n"
                 "Usage: " << pname << " [options...] <kernel type> <precisions> <layouts>\n"
                 "\n"
                 "The required arguments (<kernel type>, <precisions>, <layouts>) specify the type of\n"
                 "  kernel to search for. They follow the usual gemmstone syntax, e.g.:\n"
                 "       gemm ZZZ NTN\n"
                 "       offseta[g32] gemm [FO]OS T@8N@8N\n"
                 "\n"
                 "Multiderby requires knowledge of the GPU's main memory bandwidth, which can be specified\n"
                 "  with the MEMORY_BW=<GB/s> environment variable.\n"
                 "\n"
                 "Problem space options:\n"
                 "    -pc, --compute-bound         Only tune strategies for compute-bound problem sizes.\n"
                 "    -pm, --memory-bound          Only tune strategies for memory-bound problem sizes.\n"
                 "    -p MxNxK                     Tune strategies for the given problem size, instead of the\n"
                 "                                   default grid of sizes. Multiple -p options may be\n"
                 "                                   specified to search multiple sizes.\n"
                 "    -pf <file>                   Load problem sizes from a file. Each line of the file\n"
                 "                                   should take the form MxNxK ('x' delimiters are optional).\n"
                 "    -pr                          Periodically add randomly-selected problem sizes.\n"
                 "\n"
                 "Strategy space options:\n"
                 "    --power-2                    Only search power-of-2 tile sizes (default).\n"
                 "    --non-power-2                Also search non-power-of-2 tile sizes.\n"
                 "    --slm=A|B|-A|-B[,...]        Enable (A/B) or disable (-A/-B) search of strategies using SLM per\n"
                 "                                   operand, e.g. --slm=A,-B. By default, SLM is disallowed when the\n"
                 "                                   operand is K-major (A if T, B if N).\n"
                 "\n"
                 "Data options:\n"
                 "    -i <file>                    Load data from a prior multiderby tuning session from <file>.\n"
                 "    -o <file>                    Store data into <file>. Default: file name of the form md_*.data\n"
                 "                                   in the current directory.\n"
                 "    -k <file>                    Store top kernel selections into <file>, one per line.\n"
                 "                                   Default: kernels written to stdout when multiderby exits.\n"
                 "\n"
                 "Checkpoint options:\n"
                 "    -co <file>                   Store checkpoints to <file>. Defaults to a temporary location.\n"
                 "    -ci <file>                   Load checkpoint from the given <file>. By default multiderby\n"
                 "                                   searches for default-named checkpoints from previous sessions.\n"
                 "    -sp <time>,                  Set time interval between checkpoints (e.g. 20s, 3m, 1h).\n"
                 "      --checkpoint-period <time>   Default is 5m.\n"
                 "\n"
                 "Benchmark options:\n"
                 "    --cold                       Force cold-cache mode (default: cold-cache for memory-bound sizes,\n"
                 "                                   warm-cache for compute-bound sizes)\n"
                 "    --warm                       Force warm-cache mode\n"
                 "\n"
                 "General options:\n"
                 "    -l <time>,                   Set time limit for autotuning session (default: 2h)\n"
                 "      --time-limit <time>\n"
                 "    -cbw, --c-bandwidth          Include C matrix in bandwidth calculations\n"
                 "\n"
                 "Queries:\n"
                 "    -r MxNxK                     Get estimated roofline performance for given problem size.\n"
                 "                                   Requires data to be loaded from a previous multiderby session (-i).\n"
                 "    --dump                       Dump all PV/SV data from a checkpoint or data file (-i or -ci).\n"
                 ;
    exit(1);
}

int main(int argc, char **argv)
try {
    using namespace std::chrono;

    /* Collect options */
    auto shift = [&] {
        if (argc-- <= 0)
            throw std::runtime_error("Invalid command-line arguments");
        return *argv++;
    };
    auto unshift = [&] {
        argc++; argv--;
    };

    auto pname = shift();
    MultiderbyOptions options;
    bool interactive = true;

    while (argc > 0) {
        std::string arg = shift();
        if (arg == "--help")
            usage(pname);
        else if (arg == "-a"sv || arg == "--agents"sv)
            options.agentCount = std::atoi(shift());
        else if (arg == "-vb"sv)
            options.verboseBenchmark = true;
        else if (arg == "-vc"sv)
            options.verboseComm = true;
        else if (arg == "-vs"sv)
            options.showSensitivities = true;
        else if (arg == "-vt"sv)
            options.showTiming = true;
        else if (arg == "-pc"sv || arg == "--compute-bound"sv)
            options.doComputeBound = true;
        else if (arg == "-pm"sv || arg == "--memory-bound"sv)
            options.doMemBound = true;
        else if (arg == "-ci"sv)
            options.checkpointIn = shift();
        else if (arg == "-co"sv)
            options.checkpointOut = shift();
        else if (arg == "-i"sv)
            options.inputData = shift();
        else if (arg == "-o"sv)
            options.outputData = shift();
        else if (arg == "-k"sv)
            options.outputKernelList = shift();
        else if (arg == "-kc"sv)
            options.outputCrashedKernelList = shift();
        else if (arg == "-p"sv)
            options.problems.push_back(MultiderbyClient::stringToPV(shift()));
        else if (arg == "-pf"sv || arg == "--problems"sv) {
            std::string path = shift();
            std::ifstream pfs(path);
            if (!pfs) throw std::runtime_error("Could not read file " + path);
            for (std::string line; std::getline(pfs, line); )
                if (!line.empty() && line[0] != '#')
                    options.problems.push_back(MultiderbyClient::stringToPV(line));
        } else if (arg == "-pr"sv)
            options.addRandomProblems = true;
        else if (arg == "-s"sv)
            options.strategiesIn.push_back(shift());
        else if (arg == "-sf"sv || arg == "--strategies-in"sv) {
            std::string path = shift();
            std::ifstream sfs(path);
            if (!sfs) throw std::runtime_error("Could not read file " + path);
            for (std::string line; std::getline(sfs, line); )
                if (!line.empty() && line[0] != '#')
                    options.strategiesIn.push_back(line);
        } else if (arg == "-sp"sv || arg == "--checkpoint-period"sv)
            options.checkpointPeriod = parseTime(shift());
        else if (arg == "-l"sv || arg == "--time-limit"sv)
            options.timeLimit = parseTime(shift());
        else if (arg == "--cold"sv)
            options.coldCache = true;
        else if (arg == "--warm"sv)
            options.coldCache = false;
        else if (arg == "-cbw"sv || arg == "--c-bandwidth"sv)
            options.includeCBandwidth = true;
        else if (arg == "--power-2"sv)
            options.nonPowerOf2Tiles = false;
        else if (arg == "--non-power-2"sv)
            options.nonPowerOf2Tiles = true;
        else if (arg.rfind("--slm="sv, 0) == 0) {
            const std::string spec = arg.substr(6);
            for (size_t i = 0; i < spec.size(); ++i) {
                bool allow = true;
                if (spec[i] == '-') {
                    allow = false;
                    ++i;
                }
                if (i >= spec.size())
                    throw std::runtime_error("Missing operand for --slm option");
                switch (spec[i]) {
                    case ',':
                        continue;
                    case 'A':
                        options.slmA = allow;
                        break;
                    case 'B':
                        options.slmB = allow;
                        break;
                    default:
                        throw std::runtime_error("Unknown --slm option: " + spec);
                }
            }
        }
        else if (arg == "-r"sv) {
            options.problems.push_back(MultiderbyClient::stringToPV(*argv));
            options.rooflineProblems.push_back(MultiderbyClient::stringToPV(shift()));
            interactive = false;
        } else if (arg == "--dump"sv) {
            options.dumpData = true;
            interactive = false;
        } else if (arg[0] == '-')
            throw std::runtime_error("Invalid command-line option " + arg);
        else {
            unshift();
            break;
        }
    }

    /* Collect stem */
    if (argc < 3)
        usage(pname);

    std::string stem;
    while (argc > 0) {
        if (!stem.empty()) stem += ' ';
        stem += shift();
    }

    /* Set default options */
    if (options.checkpointOut.empty() && options.runOptimizer) {
        options.checkpointOut = options.checkpointIn.empty() ? defaultCheckpointPath(stem)
                                                             : options.checkpointIn;
    }
    if (options.checkpointPeriod <= 0)
        options.checkpointOut.clear();

    if (options.outputData.empty())
        options.outputData = defaultOutputPath(stem);

    if (options.timeLimit == 0)
        options.timeLimit = 2 * 60 * 60;

    if (!options.checkpointOut.empty()) {
        std::cerr << "Writing checkpoints every " << formatTime(options.checkpointPeriod)
                  << " to " << options.checkpointOut << std::endl;
    }

    /* Setup */
    utils::setenv("FORCE_ZEBIN", "1", false);       /* use zebin by default for faster kernel generation */
    utils::setenv("FRONTEND_TIMEOUT_MS", "1000", false);  /* set a timeout on kernel generation */
    utils::setenv("ENABLE_METRICS", "0", false);    /* multiderby doesn't need MDAPI enabled */
    interactive = interactive && terminalSetup();

    std::cerr << "Initializing...\n";

    /* Create optimizer client */
    MultiderbyClient client(stem, options);

    /* Parse incoming strategies */
    std::vector<StrategyVector> initialSVs;
    for (auto &strategy: options.strategiesIn)
        initialSVs.push_back(client.stringToSV(strategy));

    /* Generate initial grid and create optimizer */
    auto &&initialPVs = !options.problems.empty() ? options.problems
                                                  : client.defaultPVGrid();
    MultiderbyOptimizer optimizer(client, options, initialPVs, initialSVs);

    /* Load a checkpoint if supplied */
    bool dataIn = false;
    if (!options.checkpointIn.empty()) {
        multiopt::Serialized s(readFile(options.checkpointIn));
        s >> optimizer;
        dataIn = true;
    } else if (!options.inputData.empty()) {
        multiopt::Serialized s(readFile(options.inputData));
        s >> optimizer;
        dataIn = true;
    } else if (interactive) {
        auto cp = defaultCheckpointPath(stem);
        if (std::error_code ec; exists(cp, ec)) {
            std::cerr << "Found existing checkpoint file at " << cp << ". Restore from checkpoint (y/N)? \n";
            if (std::tolower(terminalInput(true)) == 'y') {
                multiopt::Serialized s(readFile(cp));
                s >> optimizer;
            }
        }
    }

    /* Roofline inquiries */
    if (!options.rooflineProblems.empty()) {
        if (!dataIn)
            throw std::runtime_error("Checking rooflines requires data from a previous multiderby session (-i option)");
        optimizer.updateTopKernels();
        auto svs = optimizer.getTopKernels();
        for (auto &pv_unlabeled: options.rooflineProblems) {
            // Hack: Don't pull the problem vector from the data store, because it may not exist.
            // Multiderby -> benchstone shouldn't care either way
            auto pv = multiopt::Labeled<ProblemVector>();
            pv.vec = pv_unlabeled;
            using stratPerf = std::pair<std::string, float>;
            std::vector<stratPerf> svPerf;
            for (auto &sv : svs) {
                multiopt::AgentRequest<ProblemVector, StrategyVector> req;
                req.sv = sv;
                req.pvs = {pv};
                client.benchmark(req);
                if (req.perf.empty()) {
                    continue;
                }
                float perf = req.perf[0].value;
                float expected = client.absRoofline(pv);
                svPerf.emplace_back(client.svToString(sv), perf/expected);
            }
            std::sort(svPerf.begin(), svPerf.end(), [](stratPerf i, stratPerf j) {
                return i.second > j.second;
            });

            int max_strategies = 10;
            for (auto &perf : svPerf) {
                if (max_strategies-- == 0) break;
                std::cout << perf.first << " on " << MultiderbyClient::pvToString(pv) << ": " << perf.second * 100 << "%" << std::endl;
            }
            std::cout << optimizer.roofline(pv) << std::endl;
        }
        return 0;
    }

    /* Data dump */
    if (options.dumpData) {
        if (!dataIn)
            throw std::runtime_error("--dump requires data from a previous multiderby session (-i or -ci option)");
        std::cout << optimizer.dump();
        return 0;
    }

    /* Main loop */
    auto start = steady_clock::now();
    auto lastCheckpoint = start;
    bool run = true;

    double refreshInterval = MultiderbySettings::maxRefreshInterval;
    while (run) {
        optimizer.run(refreshInterval);

        auto now = steady_clock::now();
        auto elapsed = (now - start) / seconds(1);
        auto elapsedCP = (now - lastCheckpoint) / seconds(1);

        if (options.timeLimit > 0 && elapsed > options.timeLimit) {
            std::cerr << "\nReached time limit.\n";
            break;
        }

        if (!options.checkpointOut.empty() && elapsedCP >= options.checkpointPeriod) {
            writeFile(options.checkpointOut, multiopt::serialize(optimizer));
            lastCheckpoint = now;
        }

        auto renderStart = steady_clock::now();

        std::ostringstream frame;
        frame << VT_GREEN "multiderby" VT_RESET " is running... ";
        if (options.timeLimit > elapsed)
            frame << '(' << formatTime(options.timeLimit - elapsed) << " remaining)";
        frame << "\n\n";
        frame << optimizer.status() << "\n\n";
        if (options.showTiming)
            frame << optimizer.timings() << "\n\n";
        if (options.showSensitivities)
            frame << optimizer.sensitivities() << "\n\n";
        frame << optimizer.kernelList();
        if (auto &crashed = client.crashedStrategyList(); !crashed.empty())
            frame << "\n" VT_BOLD "Note: " VT_RESET << crashed.size() << " strategies crashed during benchmarking.\n";
        frame << "\nPress " VT_BOLD "q" VT_RESET " to quit, " VT_BOLD "s" VT_RESET " to update kernel selections.\n";

        std::cerr << '\r' << VT_CLEAR_SCREEN << frame.str();
        auto renderElapsed = (steady_clock::now() - renderStart) / std::chrono::duration<double>(1);

        // Retune refresh rate for the next iteration based on measured render cost.
        refreshInterval = std::clamp(renderElapsed / MultiderbySettings::targetRenderOverhead,
                                     MultiderbySettings::minRefreshInterval, MultiderbySettings::maxRefreshInterval);

        /* Check for input commands */
        if (interactive) switch (std::tolower(terminalInput())) {
            case '>': options.verboseComm = !options.verboseComm; break;
            case 'q': run = false; break;
            case 's': optimizer.updateTopKernels(); break;
            default: break;
        }
    }

    /* Finish outstanding tasks, redrawing periodically so the drain doesn't
     * look like it has hung (the final spreading pass can take a while). */
    optimizer.freeze();
    optimizer.updateTopKernels();   // queue top-kernel spreading so the drain below selects a final list
    refreshInterval = MultiderbySettings::minRefreshInterval;
    while (optimizer.hasPendingWork()) {
        optimizer.run(refreshInterval);

        auto renderStart = steady_clock::now();

        std::ostringstream frame;
        frame << VT_GREEN "multiderby" VT_RESET " is finishing up...\n\n";
        frame << optimizer.status() << "\n\n";

        std::cerr << '\r' << VT_CLEAR_SCREEN << frame.str();
        auto renderElapsed = (steady_clock::now() - renderStart) / std::chrono::duration<double>(1);
        refreshInterval = std::clamp(renderElapsed / MultiderbySettings::targetRenderOverhead,
                                     MultiderbySettings::minRefreshInterval, MultiderbySettings::maxRefreshInterval);
    }
    std::cerr << '\r' << VT_CLEAR_SCREEN;
    std::cerr << "Finishing up... done.\n";

    /* Switch back to the main screen before the final dump below, so it
     * lands in normal scrollback instead of being wiped with the alt screen. */
    if (interactive)
        terminalTeardown();

    std::cerr << optimizer.kernelList() << std::endl;

    /* Write out data if requested. */
    if (!options.outputData.empty()) {
        std::cerr << "Data saved to: " << options.outputData << std::endl;
        writeFile(options.outputData, multiopt::serialize(optimizer));

        /* Remove checkpoint file if successful. */
        if (run && !options.checkpointOut.empty()) {
            std::error_code ec;
            std::filesystem::remove(options.checkpointOut, ec);
        }
    }

    if (auto &crashed = client.crashedStrategyList(); !crashed.empty()) {
        if (!options.outputCrashedKernelList.empty()) {
            std::cerr << "Crashed kernels saved to: " << options.outputCrashedKernelList << std::endl;
            writeFile(options.outputCrashedKernelList, crashed);
        }
    }

    if (optimizer.haveKernelList()) {
        if (!options.outputKernelList.empty()) {
            std::cerr << "Final kernels saved to: " << options.outputKernelList << std::endl;
            writeFile(options.outputKernelList, optimizer.finalKernelList());
        } else {
            std::cerr << "Final kernels:\n";
            std::cout << optimizer.finalKernelList();
        }
    }
} catch (std::exception &e) {
    // Leave the alternate terminal before printing the error message.
    terminalTeardown();
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
