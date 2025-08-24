cmake_minimum_required(VERSION 3.16...3.22)

find_program(CLANGPP "clang++")
set(CMAKE_CXX_COMPILER ${CLANGPP} CACHE STRING "C++ compiler" FORCE)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)


project (
    batched_cc
    VERSION 0.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)



# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo")  # enable line info in ncu ui


### Begin vamp compiler options
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
	# Need explicit AVX2 for some MacOS clang versions
	set(VAMP_ARCH "-march=native -mavx2")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
	# ARM platforms (aarch64 / arm64)
	set(VAMP_ARCH "-mcpu=native -mtune=native")
else()
	message(FATAL_ERROR "Unsupported architecture ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# default fast args that work on all platforms
set(VAMP_FAST_ARGS "-fno-math-errno -fno-signed-zeros -fno-trapping-math -fno-rounding-math -ffp-contract=fast")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") # x86 supports additional flags
	string(APPEND VAMP_FAST_ARGS " -fassociative-math")
	if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # Clang supports additional fine-grained flags over GCC
		string(APPEND VAMP_FAST_ARGS " -fno-honor-infinities -fno-honor-nans")
		if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 11.0.0) # clang 11 does not support -fapprox-func
			string(APPEND VAMP_FAST_ARGS " -fapprox-func")
		endif()
	endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VAMP_ARCH} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 ${VAMP_FAST_ARGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VAMP_ARCH} -Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 ${VAMP_FAST_ARGS}")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
	# Valgrind can't handle avx512 instructions
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -mno-avx512f")
	set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -mno-avx512f")
endif()

if(VAMP_LTO)
	if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
		set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -flto")
	elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		# specify auto paralellism for gcc linking
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto=auto")
		set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -flto=auto")
	endif()
endif()
### end vamp compiler options

### motion benchmark script
add_executable(evaluate_mbm
    scripts/evaluate_mbm.cpp
    # src/planning/pRRT.cu
    # src/planning/nRRT.cu
    src/planning/pRRTC.cu
    # src/planning/pwRRTC.cu
    # src/planning/nRRTC.cu
)

target_include_directories(evaluate_mbm
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/vamp/src/impl/vamp
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/vamp/src/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/nigh-src/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/pdqsort
)


target_link_libraries(evaluate_mbm
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)

target_compile_options(evaluate_mbm PRIVATE
    -Wno-c++11-narrowing
)

### single_mbm
add_executable(single_mbm
    scripts/single_mbm.cpp
    # src/planning/pRRT.cu
    # src/planning/nRRT.cu
    # src/planning/nRRTC.cu
    src/planning/pRRTC.cu
    # src/planning/pwRRTC.cu

)

target_include_directories(single_mbm
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/vamp/src/impl/vamp
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/vamp/src/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/nigh-src/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/pdqsort
)

target_link_libraries(single_mbm
    PRIVATE
    # template_lib
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)

target_compile_options(single_mbm PRIVATE
    -Wno-c++11-narrowing
)

### validate_configs
add_executable(validate_configs
    scripts/validate_configs.cpp
)

target_include_directories(validate_configs
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/vamp/src/impl/vamp
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/vamp/src/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/nigh-src/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/pdqsort
)

target_link_libraries(validate_configs
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)

target_compile_options(validate_configs PRIVATE
    -Wno-c++11-narrowing
)
