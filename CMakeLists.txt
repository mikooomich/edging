cmake_minimum_required(VERSION 3.10)
project(edging)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(DEFINED ENV{CMAKE_TOOLCHAIN_FILE})
  set(CMAKE_TOOLCHAIN_FILE "$ENV{CMAKE_TOOLCHAIN_FILE}"
      CACHE STRING "")
endif()


add_executable(main main.cpp
        sequential.cpp
        bitmap.cpp
        utils.cpp
)
