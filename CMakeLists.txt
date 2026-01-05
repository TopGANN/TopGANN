cmake_minimum_required(VERSION 3.5)
project(AGRENN)
# add_compile_options(-Wall -Wextra -Werror)
add_compile_options(-w)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -mavx -mavx2 -msse3  -fopenmp ")

include_directories(/home/ilias.azizi/projects/AGRENN/include)


add_executable(AGRENN main.cpp)
target_link_libraries(AGRENN)