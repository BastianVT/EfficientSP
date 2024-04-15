cmake_minimum_required(VERSION 3.13)  # CMake version check
project(pctsp)               # Create project "pctsp"
set(CMAKE_CXX_STANDARD 17)            # Enable c++17 standard


#add_executable(pctsp
#        argparse.cpp
#        local_search_with_loss.cpp)


add_executable(pctsp
        main.cpp
        argparse.cpp
        Params.h
        Params.cpp
        Params_WL.h
        Params_WL.cpp
        Params_WL.h
        Params_WL.cpp
        Solution.h
        Solution.cpp
        Solution_WL.h
        Solution_WL.cpp
        ILS.h
        ILS.cpp
        ILS_WL.h
        ILS_WL.cpp
        utils.h
        utils.cpp)

