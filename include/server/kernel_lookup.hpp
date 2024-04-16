#include "types.hpp"

map<string, const void *> &lookup();

using KernelAccessLoc = map<const void*, ordered_set<pair<int, int>>>;
map<string, KernelAccessLoc> &kernel_model_access_loc();