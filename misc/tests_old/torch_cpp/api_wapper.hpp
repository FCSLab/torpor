#include <map>
#include <vector>

/* register */
typedef struct {
    dim3 gridDim;
    dim3 blockDim;
    int counter;
    std::vector<std::pair<void *, size_t>> args;
} kernelInfo_t;

std::map<std::string, const void *> &lookup();