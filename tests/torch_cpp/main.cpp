#include <torch/script.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <map>
#include <set>
#include "api_wapper.hpp"
#include "list.h"

#include <string> 
#include <sstream>
#include <fstream> 

// extern std::map<std::string, const void *> &lookup();

void check_symbol_addr(const char* query_string) {
    if (lookup().find(std::string(query_string)) != lookup().end()) {
        auto ptr = lookup()[std::string(query_string)];
        std::cout << "Func addr of " << query_string << " : " << (uint64_t) ptr << std::endl;
    }
    else{
        std::cout << "Not found func addr of " << query_string << std::endl;
    }
    // void* ptr;
    // auto err = cudaGetSymbolAddress(&ptr, query_string);
    // std::cout << "Try get symbol " << query_string << " err " << std::to_string((int)err) << " addr " << (uint64_t) ptr << std::endl;
}

typedef struct kernel_info {
    char *name;
    size_t param_size;
    size_t param_num;
    uint16_t *param_offsets;
    uint16_t *param_sizes;
    void *host_fun;
} kernel_info_t;

int cpu_utils_launch_child(const char *file, char **args)
{
    int filedes[2];
    FILE *fd = NULL;

    if (pipe(filedes) == -1) {
        return -1;
    }

    pid_t pid = fork();
    if (pid == -1) {
        return -1;
    } else if (pid == 0) {
        while ((dup2(filedes[1], STDOUT_FILENO) == -1) && (errno == EINTR)) {}
        close(filedes[1]);
        close(filedes[0]);
        char *env[] = {NULL};
        execvpe(file, args, env);
        exit(1);
    }
    close(filedes[1]);
    return filedes[0];
}

static int cpu_utils_read_pars(kernel_info_t *info, FILE* fdesc)
{
    static const char* attr_str[] = {"EIATTR_KPARAM_INFO",
        "EIATTR_CBANK_PARAM_SIZE",
        "EIATTR_PARAM_CBANK"};
    enum attr_t {KPARAM_INFO = 0,
        CBANK_PARAM_SIZE = 1,
        PARAM_CBANK = 2,
        ATTR_T_LAST}; // states for state machine
    char *line = NULL;
    size_t linelen = 0;
    int ret = 1;
    int read = 0;
    char key[32];
    char val[256] = {0};
    size_t val_len = 0;
    enum attr_t cur_attr = ATTR_T_LAST; // current state of state machine
    int consecutive_empty_lines = 0;
    info->param_num = 0;
    info->param_offsets = NULL;
    info->param_sizes = NULL;
    while (getline(&line, &linelen, fdesc) != -1) {
        memset(val, 0, 256);
        read = sscanf(line, "%31s %255c\n", key, val);
        val_len = strlen(val);
        if (val_len > 0) {
            val[strlen(val)-1] = '\0';
        }
        if (read == -1 || read == 0) {
            if (++consecutive_empty_lines >= 2) {
                break; //two empty line means there is no more info for this kernel
            } else {
                continue;
            }
        } else {
            consecutive_empty_lines = 0;
            if (read == 1) {
                continue; // some lines have no key-value pair.
                // We are not interested in those lines.
            }
        }
        if (strcmp(key, "Attribute:") == 0) { // state change
            cur_attr = ATTR_T_LAST;
            for (int i=0; i < ATTR_T_LAST; i++) {
                if (strcmp(val, attr_str[i]) == 0) {
                    cur_attr = (attr_t) i;
                }
            }
        } else if(strcmp(key, "Value:") == 0) {
            size_t buf;
            uint16_t ordinal, offset, size;
            switch(cur_attr) {
            case KPARAM_INFO:
                if (sscanf(val, "Index : 0x%*hx Ordinal : 0x%hx Offset : 0x%hx Size : 0x%hx\n", &ordinal, &offset, &size) != 3 ) {
                    goto cleanup;
                }
                if (ordinal >= info->param_num) {
                    info->param_offsets = (uint16_t*) realloc(
                                                  info->param_offsets,
                                                  (ordinal+1)*sizeof(uint16_t));
                    info->param_sizes = (uint16_t*) realloc(
                                                (void*)info->param_sizes,
                                                (ordinal+1)*sizeof(uint16_t));
                    info->param_num = ordinal+1;
                }
                info->param_offsets[ordinal] = offset;
                info->param_sizes[ordinal] = size;
                break;
            case CBANK_PARAM_SIZE:
                if (sscanf(val, "0x%lx", &info->param_size) != 1) {
                    goto cleanup;
                }
                break;
            case PARAM_CBANK:
                if (sscanf(val, "0x%*x 0x%lx", &buf) != 1) {
                    goto cleanup;
                }
                break;
            default:
                break;
            }
        }


    }

    ret = 0;
 cleanup:
    free(line);
    return ret;
}

int cpu_utils_parameter_info(list *kernel_infos, char *path)
{
    int ret = 1;
    char linktarget[PATH_MAX] = {0};
    char *args[] = {"/usr/local/cuda/bin/cuobjdump", "--dump-elf", NULL, NULL};
    int output;
    FILE *fdesc; //fd to read subcommands output from
    int child_exit = 0;
    char *line = NULL;
    size_t linelen;
    static const char nv_info_prefix[] = ".nv.info.";
    kernel_info_t *buf = NULL;
    char *kernelname;
    // struct stat filestat = {0};

    std::ofstream output_file;
    output_file.open ("kernel_info.txt");

    if (kernel_infos == NULL) {
        std::cout << "list is NULL.\n";
        goto out;
    }

    // if (stat(path, &filestat) != 0) {
    //     std::cout << "stat on failed.\n";
    //     goto out;
    // }

    // if (S_ISLNK(filestat.st_mode) || strcmp(path, "/proc/self/exe") == 0) {
    //     if (readlink("/proc/self/exe", linktarget, PATH_MAX) == PATH_MAX) {
    //         goto out;
    //     }
    //     args[2] = linktarget;
    // } else {
    //     args[2] = path;
    // }
    args[2] = path;
    std::cout << "searching for kernels in " << args[2] << std::endl;


    if ( (output = cpu_utils_launch_child(args[0], args)) == -1) {
        std::cout << "error while launching child." << std::endl;
        goto out;
    }

    if ( (fdesc = fdopen(output, "r")) == NULL) {
        std::cout << "error while opening stream." << std::endl;
        goto cleanup1;
    }

    while (getline(&line, &linelen, fdesc) != -1) {
        if (strncmp(line, nv_info_prefix, strlen(nv_info_prefix)) != 0) {
            // Line does not start with .nv.info. so continue searching.
            continue;
        }
        // Line starts with .nv.info.
        // Kernelname is line + strlen(nv_info_prefix)
        kernelname = line + strlen(nv_info_prefix);
        if (strlen(kernelname) == 0) {
            std::cout << "found .nv.info section, but kernelname is empty" << std::endl;
            goto cleanup2;
        }

        if (list_append(kernel_infos, (void**)&buf) != 0) {
            std::cout << "error while appending list." << std::endl;
            goto cleanup2;
        }

        if ((buf->name = (char*) malloc(strlen(kernelname))) == NULL) {
            std::cout << "malloc failed." << std::endl;
            goto cleanup2;
        }
        //copy string and remove trailing \n
        strncpy(buf->name, kernelname, strlen(kernelname)-1);
        buf->name[strlen(kernelname)-1] = '\0';

        if (cpu_utils_read_pars(buf, fdesc) != 0) {
            std::cout << "reading paramter infos failed." << std::endl;
            goto cleanup2;
        }
        std::cout << "found kernel " << buf->name << " param_num: " << buf->param_num << std::endl;
        output_file << buf->name;
        output_file << ",";
        output_file << buf->param_num;
        for (int i = 0; i < buf->param_num; i++) {
            output_file << ",";
            output_file << buf->param_sizes[i];
        }
        output_file << "\n";
        
        // if (concerned_symbols.find(std::string(buf->name)) != concerned_symbols.end()) {
        //     concerned_symbols.erase(std::string(buf->name));
        //     if (concerned_symbols.size() == 0) {
        //         std::cout << "all concerned kernel found " << std::endl;
        //         goto cleanup2;
        //     }
        // }

    }

    if (ferror(fdesc) != 0) {
        std::cout << "file descriptor shows an error." << std::endl;
        goto cleanup2;
    }

    ret = 0;
 cleanup2:
    fclose(fdesc);
 cleanup1:
    close(output);
    // wait(&child_exit);
 out:
    free(line);
    output_file.close();
    return (ret != 0 ? ret : child_exit);
}

list kernel_infos = {0};
// std::set<std::string> concerned_symbols;

int main(int argc, const char* argv[]){

    // torch::Device device(torch::kCUDA);
    // torch::jit::script::Module module = torch::jit::load(argv[1]);
    // module.to(device);

    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1, 3, 224, 224}).to(device));

    // at::Tensor output = module.forward(inputs).toTensor();
    // std::cout << output.slice(/*dims=*/1, /*start=*/0, /*end=*/5) << '\n';

    std::cout << "ok\n";

    // std::ifstream input(argv[2]);
    // for( std::string eachLine; getline( input, eachLine ); ){
    //     concerned_symbols.insert(eachLine);
    // }


    // check_symbol_addr("_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_49_GLOBAL__N__d9608136_16_TensorCompare_cu_d0af11f728clamp_min_scalar_kernel_implERNS_14TensorIteratorEN3c106ScalarEENKUlvE_clEvENKUlvE6_clEvEUlfE_NS_6detail5ArrayIPcLi2EEEEEviT0_T1_");

    // if (cpu_utils_parameter_info(&kernel_infos, "/opt/conda/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so") != 0) {
    if (cpu_utils_parameter_info(&kernel_infos, const_cast<char*>(argv[1])) != 0) {
        std::cout << "could not get kernel infos." << std::endl;
    }
    else {
        std::cout << "got kernel infos." << std::endl;
    }
}