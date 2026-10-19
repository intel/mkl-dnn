/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#define CL_TARGET_OPENCL_VERSION 210

#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#define MAX_PLATFORMS 16
#define MAX_DEVICES 16
#define MAX_NAME_LEN 256

void usage(const char *pname)
{
    fprintf(stderr, "Usage: %s file.cl ['flags' [plat# [device#]]] > out.bin\n", pname);
    exit(2);
}

void handle(cl_int result)
{
    if (result) {
        fprintf(stderr, "OpenCL error: %d\n", (int) result);
        exit(result);
    }
}

void async_cl_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    fprintf(stderr, "Asynchronous OpenCL error: %s.\n", errinfo);
}

int main(int argc, char **argv)
{
    cl_platform_id plat_ids[MAX_PLATFORMS], plat = NULL;
    cl_device_id dev_ids[MAX_DEVICES], dev = NULL;
    cl_uint nplat, ndev;
    int sel_plat_idx = -1, sel_dev_idx = -1;
    const char *source_fname;
    const char *flags = "";

    if (argc <= 1) usage(argv[0]);

    source_fname = argv[1];
    if (argc > 2) flags = argv[2];
    if (argc > 3) sel_plat_idx = atoi(argv[3]);
	if (argc > 4) sel_dev_idx  = atoi(argv[4]);

    char plat_name[MAX_NAME_LEN], dev_name[MAX_NAME_LEN];

    handle(clGetPlatformIDs(MAX_PLATFORMS, plat_ids, &nplat));
    if (sel_plat_idx >= nplat) {
        fprintf(stderr, "Invalid platform index %d: only %d platforms available.\n", sel_plat_idx, nplat);
        return 1;
    }
    if (nplat == 0) {
        fprintf(stderr, "No OpenCL platforms are available.\n");
        return 1;
    }

    for (int plat_idx = 0; plat_idx < nplat; plat_idx++) {
        if (sel_plat_idx >= 0 && sel_plat_idx != plat_idx)
            continue;

        plat = plat_ids[plat_idx];

        cl_int status = clGetDeviceIDs(plat, sel_dev_idx >= 0 ? CL_DEVICE_TYPE_ALL : CL_DEVICE_TYPE_GPU, MAX_DEVICES, dev_ids, &ndev);

        if (status == CL_DEVICE_NOT_FOUND)
            ndev = 0;
        else
            handle(status);

        if (sel_dev_idx >= ndev) {
            fprintf(stderr, "Invalid device index %d: only %d devices available in platform %d.\n", sel_dev_idx, ndev, plat_idx);
            return 1;
        } else if (sel_dev_idx < 0) {
            if (ndev <= 0)
                continue;
            sel_dev_idx = 0;
            sel_plat_idx = plat_idx;
        }

        dev = dev_ids[sel_dev_idx];
        break;
    }

    if (!dev || !plat) {
        fprintf(stderr, "Could not locate any GPUs. Please specify platform and device numbers.\n");
        return 1;
    }

    handle(clGetPlatformInfo(plat, CL_PLATFORM_NAME, MAX_NAME_LEN, plat_name, NULL));
    handle(clGetDeviceInfo(dev, CL_DEVICE_NAME, MAX_NAME_LEN, dev_name, NULL));
    plat_name[MAX_NAME_LEN - 1] = '\0';
    dev_name[MAX_NAME_LEN - 1] = '\0';

    cl_context ctx = clCreateContext(NULL, 1, &dev, &async_cl_notify, NULL, NULL);
    if (!ctx) {
        fprintf(stderr, "Context creation failed.\n");
        return 1;
    }

    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, dev, NULL, NULL);
    if (!queue) {
        fprintf(stderr, "Queue creation failed.\n");
        return 1;
    }

    FILE *bfile = fopen(source_fname, "r");
    if (!bfile) {
        fprintf(stderr, "Could not open source file %s\n", source_fname);
        return 1;
    }

    fseek(bfile, 0L, SEEK_END);
    size_t source_length = ftell(bfile);

    rewind(bfile);

    char *source = malloc(source_length);
    if (!source) {
        fprintf(stderr, "Could not allocate memory buffer for reading source.\n");
        return 1;
    }

    if (source_length != fread(source, 1, source_length, bfile)) {
        fprintf(stderr, "Could not read source file %s\n", source_fname);
        return 1;
    }

    cl_int err = 0;
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char **) &source, &source_length, &err);
    if (!program || err) {
        fprintf(stderr, "Program creation failed: error %d.\n", (int) err);
        return 1;
    }

    cl_int result = clBuildProgram(program, 0, NULL, flags, NULL, NULL);
    if (result == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;

        handle(clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));

        char *log = malloc(log_size+1);

        handle(clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));

        log[log_size] = 0;
        fprintf(stderr, "Build failed.\n%s", log);

        exit(11);
    } else
        handle(result);

    size_t binary_size = 0;
    handle(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL));

    unsigned char *binary = malloc(binary_size);
    handle(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binary), &binary, NULL));

#ifdef _WIN32
    if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
        fprintf(stderr, "Error setting binary mode on stdout.\n");
        return 1;
    }
#endif

    if (binary_size != fwrite(binary, 1, binary_size, stdout)) {
        fprintf(stderr, "Error writing binary.\n");
        return 1;
    }

    handle(clReleaseProgram(program));
    handle(clReleaseCommandQueue(queue));
    handle(clReleaseContext(ctx));
}
