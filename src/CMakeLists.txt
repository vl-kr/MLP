# CMakeList.txt : CMake project for main, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("main")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp -D_GLIBCXX_PARALLEL -Ofast -std=c++17 -march=native")

# Add source to this project's executable.
add_executable (main 
"main.cpp"
"main.h"

"libs/matrix/matrix.cpp"
"libs/matrix/matrix.h"

"libs/io/io.cpp"
"libs/io/io.h"

 "libs/network/network.cpp" 
 "libs/network/network.h"
 
 "libs/util/util.cpp" 
 "libs/util/util.h")

# TODO: Add tests and install targets if needed.
