#ifndef INCLUDE_TYPES_HPP_
#define  INCLUDE_TYPES_HPP_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include "spdlog/spdlog.h"

using string = std::string;

template <class T>
using vector = std::vector<T>;

template <class T>
using queue = std::queue<T>;

template <class T>
using set = std::unordered_set<T>;

template <class T>
using ordered_set = std::set<T>;

template <class K, class V>
using map = std::unordered_map<K, V>;

template <class K, class V>
using ordered_map = std::map<K, V, std::greater<>>;

template <class F, class S>
using pair = std::pair<F, S>;

using logger = std::shared_ptr<spdlog::logger>;

#endif // INCLUDE_TYPES_HPP_