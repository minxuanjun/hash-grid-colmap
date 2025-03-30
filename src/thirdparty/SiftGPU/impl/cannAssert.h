//
// Created by minxuan on 11/23/24.
//

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

// glog
#include <glog/logging.h>
// fmt
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>


#define CANN_ASSERT(X)                                       \
  do {                                                       \
    if (!(X)) {                                              \
      fprintf(stderr,                                        \
              "Cann Assertion '%s' failed in %s at %s:%d\n", \
              #X,                                            \
              __PRETTY_FUNCTION__,                           \
              __FILE__,                                      \
              __LINE__);                                     \
      abort();                                               \
    }                                                        \
  } while (false)

#define CANN_ASSERT_MSG(X, MSG)   \
  do {                            \
    if (!(X)) {                   \
      std::cerr << (MSG) << '\n'; \
      abort();                    \
    }                             \
  } while (false)

#define CANN_ASSERT_FMT(X, FMT, ...)                              \
  do {                                                            \
    if (!(X)) {                                                   \
      std::cerr << fmt::format(FMT, ##__VA_ARGS__) << '\n';       \
      abort();                                                    \
    }                                                             \
  } while (false)
