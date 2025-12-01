cmake_minimum_required(VERSION 3.15)
project(pylib_wrapper_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# 1. 引入 pybind11
add_subdirectory(external/pybind11)

# 2. 【核心步骤】添加子目录
# 这会触发 external/cpp_algo/CMakeLists.txt 的执行
# 并把目标 cpp_algo 暴露给当前作用域
add_subdirectory(external/NeuSim)

# 3. 创建 Python 扩展模块
# 模块名必须和 bindings.cpp 以及 __init__.py 里的引用一致
pybind11_add_module(sim src/bindings.cpp)

# 4. 链接
# 将 pybind 模块与独立的 C++ 库链接
# 这里的 cpp_algo 就是子目录中定义的那个 target
target_link_libraries(sim PRIVATE neu_sim)

# 获取当前 CMakeLists.txt 所在的目录
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 强制将生成的库文件放到 Python 包的源代码目录下
# 假设你的 python 包文件夹叫 pylib_wrapper
set_target_properties(sim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${ROOT_DIR}/soul"
    # Windows 还需要设置 Runtime
    RUNTIME_OUTPUT_DIRECTORY "${ROOT_DIR}/soul"
)
