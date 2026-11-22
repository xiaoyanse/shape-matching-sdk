cmake_minimum_required(VERSION 3.21)
project(VisionForgeShapeMatching VERSION 0.4.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(VF_SHAPE_AVX2 "Build an experimental AVX2-only kernel" OFF)
option(VF_SHAPE_PARALLEL_WINDOWS "Evaluate two independent refinement windows concurrently" ON)
option(VF_SHAPE_PROFILE_STAGES "Emit per-call diagnostic timings and retry counters" OFF)
option(VF_SHAPE_CUDA "Build the CUDA backend (W1 skeleton: residency + frame upload; static cudart)" OFF)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui)
find_package(Threads REQUIRED)
if(VF_SHAPE_CUDA)
    enable_language(CUDA)
    # 加载方案（gpu-port-phase0 §二拍板·甲）：单 DLL + 静态 cudart，免运行时再分发
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # 三代 fatbin 仅嵌 SASS 无 PTX（沿尖峰口径：跨更代架构需重编）
    set(CMAKE_CUDA_ARCHITECTURES 86-real;89-real;120-real)
    # 坑表（尖峰 README）：宿主 MSVC 源字符集须经 -Xcompiler 传递；
    # .cu 字符串字面量保持 ASCII（nvcc 前端按系统代码页词法分析）
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
set(VF_GPU_SOURCES)
if(VF_SHAPE_CUDA)
    list(APPEND VF_GPU_SOURCES src/gpu/gpu_device.cu src/gpu/coarse_kernels.cu
        src/gpu/fine_windows.cu src/gpu/gpu_backend.cpp src/gpu/coarse_extract.cpp
        src/gpu/coarse_service.cpp src/gpu/fine_service.cpp)
endif()
add_library(visionforge_shape SHARED
    src/abi.cpp src/model_codec.cpp src/matcher.cpp src/polygon.cpp src/two_stage.cpp
    src/two_stage_train.cpp src/container_codec.cpp src/geometric_refinement.cpp
    src/audit.cpp src/coarse_image.cpp src/rescue_score.cpp
    vendor/shape_based_matching/line2Dup.cpp
    ${VF_GPU_SOURCES})
target_include_directories(visionforge_shape PRIVATE include src ${OpenCV_INCLUDE_DIRS}
    vendor/shape_based_matching vendor/shape_based_matching/MIPP)
target_link_libraries(visionforge_shape PRIVATE ${OpenCV_LIBS} Threads::Threads)
set_target_properties(visionforge_shape PROPERTIES CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES INSTALL_RPATH "$ORIGIN")
if(MSVC)
    # COMPILE_LANGUAGE 限定：MSVC 裸选项不得透传 nvcc（会被当作输入文件）
    target_compile_options(visionforge_shape PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3> $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
else()
    target_compile_options(visionforge_shape PRIVATE -Wall -Wextra)
endif()
if(VF_SHAPE_AVX2)
    if(NOT MSVC)
        message(FATAL_ERROR "VF_SHAPE_AVX2 currently requires MSVC")
    endif()
    target_compile_options(visionforge_shape PRIVATE /arch:AVX2)
endif()
if(VF_SHAPE_PARALLEL_WINDOWS)
    target_compile_definitions(visionforge_shape PRIVATE VF_SHAPE_PARALLEL_WINDOWS=1)
endif()
if(VF_SHAPE_PROFILE_STAGES)
    target_compile_definitions(visionforge_shape PRIVATE VF_SHAPE_PROFILE_STAGES=1)
endif()
if(VF_SHAPE_CUDA)
    # two_stage.cpp 粗层接缝按本宏条件化；仅主库目标定义（静态编 two_stage.cpp
    # 的测试目标不接 GPU，避免 gpu 符号失链）
    target_compile_definitions(visionforge_shape PRIVATE VF_SHAPE_CUDA=1)
endif()
install(TARGETS visionforge_shape RUNTIME DESTINATION . LIBRARY DESTINATION .)
option(VF_SHAPE_BUILD_TESTS "Build native geometry regression tests" OFF)
if(VF_SHAPE_BUILD_TESTS)
    enable_testing()
    add_executable(shape_geometry_test tests/geometric_refinement_test.cpp src/geometric_refinement.cpp)
    target_include_directories(shape_geometry_test PRIVATE include src ${OpenCV_INCLUDE_DIRS}
        vendor/shape_based_matching vendor/shape_based_matching/MIPP)
    target_link_libraries(shape_geometry_test PRIVATE ${OpenCV_LIBS})
    if(MSVC)
        target_compile_options(shape_geometry_test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME shape_geometry COMMAND shape_geometry_test)
    add_executable(shape_frame_test tests/frame_preparation_test.cpp)
    target_include_directories(shape_frame_test PRIVATE ${OpenCV_INCLUDE_DIRS}
        vendor/shape_based_matching vendor/shape_based_matching/MIPP)
    target_link_libraries(shape_frame_test PRIVATE ${OpenCV_LIBS} Threads::Threads)
    if(MSVC)
        target_compile_options(shape_frame_test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME shape_frame COMMAND shape_frame_test)
    add_executable(shape_coarse_image_test tests/coarse_image_test.cpp src/coarse_image.cpp)
    target_include_directories(shape_coarse_image_test PRIVATE src ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(shape_coarse_image_test PRIVATE ${OpenCV_LIBS})
    if(MSVC)
        target_compile_options(shape_coarse_image_test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME shape_coarse_image COMMAND shape_coarse_image_test)
    add_executable(shape_clustering_test tests/clustering_ambiguity_test.cpp src/matcher.cpp
        vendor/shape_based_matching/line2Dup.cpp)
    target_include_directories(shape_clustering_test PRIVATE include src ${OpenCV_INCLUDE_DIRS}
        vendor/shape_based_matching vendor/shape_based_matching/MIPP)
    target_link_libraries(shape_clustering_test PRIVATE ${OpenCV_LIBS} Threads::Threads)
    if(MSVC)
        target_compile_options(shape_clustering_test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME shape_clustering COMMAND shape_clustering_test)
    add_executable(shape_abi_chain_test tests/abi_chain_test.cpp)
    target_include_directories(shape_abi_chain_test PRIVATE include tests ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(shape_abi_chain_test PRIVATE visionforge_shape ${OpenCV_LIBS}
        Threads::Threads)
    if(MSVC)
        target_compile_options(shape_abi_chain_test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME shape_abi_chain COMMAND shape_abi_chain_test)
    add_executable(shape_abi_tamper_test tests/abi_tamper_test.cpp)
    target_include_directories(shape_abi_tamper_test PRIVATE include tests ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(shape_abi_tamper_test PRIVATE visionforge_shape ${OpenCV_LIBS}
        Threads::Threads)
    if(MSVC)
        target_compile_options(shape_abi_tamper_test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME shape_abi_tamper COMMAND shape_abi_tamper_test)
    add_executable(shape_diagnostics_test tests/abi_diagnostics_test.cpp)
    target_include_directories(shape_diagnostics_test PRIVATE include tests ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(shape_diagnostics_test PRIVATE visionforge_shape ${OpenCV_LIBS}
        Threads::Threads)
    if(MSVC)
        target_compile_options(shape_diagnostics_test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME shape_diagnostics COMMAND shape_diagnostics_test)
    add_executable(shape_audit_test tests/audit_test.cpp)
    target_include_directories(shape_audit_test PRIVATE include tests ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(shape_audit_test PRIVATE visionforge_shape ${OpenCV_LIBS}
        Threads::Threads)
    if(MSVC)
        target_compile_options(shape_audit_test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME shape_audit COMMAND shape_audit_test)
    add_executable(shape_polygon_test tests/polygon_test.cpp)
    target_include_directories(shape_polygon_test PRIVATE include tests ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(shape_polygon_test PRIVATE visionforge_shape ${OpenCV_LIBS}
        Threads::Threads)
    if(MSVC)
        target_compile_options(shape_polygon_test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME shape_polygon COMMAND shape_polygon_test)
    add_executable(shape_polygon_mask_test tests/polygon_mask_test.cpp src/matcher.cpp
        src/polygon.cpp vendor/shape_based_matching/line2Dup.cpp)
    target_include_directories(shape_polygon_mask_test PRIVATE include src ${OpenCV_INCLUDE_DIRS}
        vendor/shape_based_matching vendor/shape_based_matching/MIPP)
    target_link_libraries(shape_polygon_mask_test PRIVATE ${OpenCV_LIBS} Threads::Threads)
    if(MSVC)
        target_compile_options(shape_polygon_mask_test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME shape_polygon_mask COMMAND shape_polygon_mask_test)
    add_executable(shape_witness_test tests/witness_test.cpp src/matcher.cpp
        src/model_codec.cpp src/coarse_image.cpp src/geometric_refinement.cpp
        src/rescue_score.cpp
        vendor/shape_based_matching/line2Dup.cpp)
    target_include_directories(shape_witness_test PRIVATE include src ${OpenCV_INCLUDE_DIRS}
        vendor/shape_based_matching vendor/shape_based_matching/MIPP)
    target_link_libraries(shape_witness_test PRIVATE ${OpenCV_LIBS} Threads::Threads)
    if(MSVC)
        target_compile_options(shape_witness_test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME shape_witness COMMAND shape_witness_test)
    # 遮挡救分：估计器/重整纯函数单测 + 清洁等价与救活集成 + 离线审计模式
    #（--v-audit 先决单调性门 / --fp-probe 假峰带采样·门槛=编译期 β·T，python 驱动调用）
    add_executable(shape_occlusion_rescue_test tests/occlusion_rescue_test.cpp
        src/rescue_score.cpp src/matcher.cpp src/two_stage.cpp src/two_stage_train.cpp
        src/geometric_refinement.cpp src/coarse_image.cpp src/polygon.cpp
        vendor/shape_based_matching/line2Dup.cpp)
    target_include_directories(shape_occlusion_rescue_test PRIVATE include src
        ${OpenCV_INCLUDE_DIRS} vendor/shape_based_matching
        vendor/shape_based_matching/MIPP)
    target_link_libraries(shape_occlusion_rescue_test PRIVATE ${OpenCV_LIBS}
        Threads::Threads)
    if(MSVC)
        target_compile_options(shape_occlusion_rescue_test PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME shape_occlusion_rescue COMMAND shape_occlusion_rescue_test)
    # Windows 下 opencv_world 运行库按 PATH 解析：ctest 环境注入 OpenCV bin，
    # 防 loader 找不到 DLL 时向桌面弹系统错误框（测试/构建产物禁止以弹窗
    # 形态触达决策人桌面——常设纪律）。bin 与 OpenCV_DIR(lib) 同级。
    set(VF_TEST_ENV "PATH=${OpenCV_DIR}/../bin\\;$ENV{PATH}")
    set(VF_TEST_NAMES shape_geometry shape_frame shape_coarse_image shape_clustering
        shape_abi_chain shape_abi_tamper shape_diagnostics shape_audit shape_polygon
        shape_polygon_mask shape_witness shape_occlusion_rescue)
    if(VF_SHAPE_CUDA)
        # W1 GPU 骨架冒烟：上下文 + Model 特征驻留 + 1080P 帧上传 + LUT 同源
        #（直编引擎源与 .cu，GPU 骨架不入公共 ABI——规划未决项 1 口径）
        add_executable(shape_gpu_residency_test tests/gpu_residency_test.cpp
            src/gpu/gpu_device.cu src/gpu/gpu_backend.cpp src/matcher.cpp src/polygon.cpp
            vendor/shape_based_matching/line2Dup.cpp)
        target_include_directories(shape_gpu_residency_test PRIVATE include src
            ${OpenCV_INCLUDE_DIRS} vendor/shape_based_matching
            vendor/shape_based_matching/MIPP)
        target_link_libraries(shape_gpu_residency_test PRIVATE ${OpenCV_LIBS}
            Threads::Threads)
        if(MSVC)
            target_compile_options(shape_gpu_residency_test PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
        endif()
        add_test(NAME shape_gpu_residency COMMAND shape_gpu_residency_test)
        list(APPEND VF_TEST_NAMES shape_gpu_residency)
    endif()
    # W2 Phase A 粗层三核对拍（独立于引擎接缝）：fastAtan2 复刻穷举 + GPU 内核
    # 对 CPU 参考双模逐位（Spike 基线/Vendor 复刻口径）。OFF 构建编译跳行 main
    # 返回 77，ctest 报 Skipped 而非缺席（防假绿：无 CUDA 不等于测试通过）；
    # 无卡环境 device_info 失败/算力集外同样 77。
    add_executable(shape_gpu_bitexact_test tests/gpu_bitexact_test.cpp
        tests/gpu_coarse_reference.cpp)
    target_include_directories(shape_gpu_bitexact_test PRIVATE include src
        ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(shape_gpu_bitexact_test PRIVATE ${OpenCV_LIBS} Threads::Threads)
    if(MSVC)
        target_compile_options(shape_gpu_bitexact_test PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
    endif()
    if(VF_SHAPE_CUDA)
        target_sources(shape_gpu_bitexact_test PRIVATE src/gpu/gpu_device.cu
            src/gpu/coarse_kernels.cu)
        target_compile_definitions(shape_gpu_bitexact_test PRIVATE VF_SHAPE_CUDA=1)
    endif()
    add_test(NAME shape_gpu_bitexact COMMAND shape_gpu_bitexact_test)
    list(APPEND VF_TEST_NAMES shape_gpu_bitexact)
    if(VF_SHAPE_CUDA)
        # W2-B 宿主候选提取/精修单测：coarse_extract 纯 C++（无 CUDA 依赖），
        # 仅 CUDA 构建注册（OFF 构建新文件不参与编译）
        add_executable(shape_coarse_extract_test tests/coarse_extract_test.cpp
            src/gpu/coarse_extract.cpp)
        target_include_directories(shape_coarse_extract_test PRIVATE include src
            ${OpenCV_INCLUDE_DIRS} vendor/shape_based_matching
            vendor/shape_based_matching/MIPP)
        target_link_libraries(shape_coarse_extract_test PRIVATE ${OpenCV_LIBS}
            Threads::Threads)
        if(MSVC)
            target_compile_options(shape_coarse_extract_test PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
        endif()
        add_test(NAME shape_coarse_extract COMMAND shape_coarse_extract_test)
        list(APPEND VF_TEST_NAMES shape_coarse_extract)
    endif()
    # W3 波1 精修窗批量评分内核对拍：GPU run_window_scores vs 宿主 LM 镜像参考
    # 逐位（行尾续读/Mat 尾取模锚）+ extract 外部窗分消费接缝等价 + 耗时 memo。
    # OFF 构建编译跳行 main 返回 77（ctest Skipped 防假绿，同 bitexact 口径）。
    add_executable(shape_fine_windows_test tests/fine_windows_test.cpp)
    target_include_directories(shape_fine_windows_test PRIVATE include src
        ${OpenCV_INCLUDE_DIRS} vendor/shape_based_matching vendor/shape_based_matching/MIPP)
    target_link_libraries(shape_fine_windows_test PRIVATE ${OpenCV_LIBS} Threads::Threads)
    if(MSVC)
        target_compile_options(shape_fine_windows_test PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
    endif()
    if(VF_SHAPE_CUDA)
        target_sources(shape_fine_windows_test PRIVATE src/gpu/gpu_device.cu
            src/gpu/coarse_kernels.cu src/gpu/fine_windows.cu src/gpu/coarse_extract.cpp)
        target_compile_definitions(shape_fine_windows_test PRIVATE VF_SHAPE_CUDA=1)
    endif()
    add_test(NAME shape_fine_windows COMMAND shape_fine_windows_test)
    list(APPEND VF_TEST_NAMES shape_fine_windows)
    set_tests_properties(shape_fine_windows PROPERTIES SKIP_RETURN_CODE 77)
    # W2-B Part5 引擎级逐位对拍+coarse_ms ≥8× 性能门：match_two_stage 全链真装配
    # （引擎源直编测试目标——DLL 不导出内部 C++ 符号），VF_SHAPE_BACKEND 进程内
    # 翻转；OFF 构建无 CUDA main 返回 77（ctest Skipped，防假绿）。
    add_executable(shape_engine_bitexact_test tests/gpu_engine_bitexact_test.cpp
        src/two_stage.cpp src/two_stage_train.cpp src/matcher.cpp src/polygon.cpp
        src/coarse_image.cpp src/rescue_score.cpp src/geometric_refinement.cpp
        vendor/shape_based_matching/line2Dup.cpp)
    target_include_directories(shape_engine_bitexact_test PRIVATE include src
        ${OpenCV_INCLUDE_DIRS} vendor/shape_based_matching
        vendor/shape_based_matching/MIPP)
    target_link_libraries(shape_engine_bitexact_test PRIVATE ${OpenCV_LIBS}
        Threads::Threads)
    if(MSVC)
        target_compile_options(shape_engine_bitexact_test PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
    endif()
    if(VF_SHAPE_CUDA)
        target_sources(shape_engine_bitexact_test PRIVATE src/gpu/gpu_device.cu
            src/gpu/coarse_kernels.cu src/gpu/fine_windows.cu src/gpu/gpu_backend.cpp
            src/gpu/coarse_extract.cpp src/gpu/coarse_service.cpp src/gpu/fine_service.cpp)
        target_compile_definitions(shape_engine_bitexact_test PRIVATE VF_SHAPE_CUDA=1)
    endif()
    add_test(NAME shape_engine_bitexact COMMAND shape_engine_bitexact_test)
    list(APPEND VF_TEST_NAMES shape_engine_bitexact)
    # W3 波3 引擎口径窗段性能门+全帧端到端实测：trace.windows_ms 全窗段 CPU vs
    # GPU（fine_service 全窗墙钟）热稳态 300×100 中位，门 ≥2.0；全帧数据行不门
    #（第三单裁决待定稿）。同 bitexact 防假绿口径：OFF 构建跳行 main 返回 77。
    add_executable(shape_engine_bench_test tests/gpu_engine_bench_test.cpp
        src/two_stage.cpp src/two_stage_train.cpp src/matcher.cpp src/polygon.cpp
        src/coarse_image.cpp src/rescue_score.cpp src/geometric_refinement.cpp
        vendor/shape_based_matching/line2Dup.cpp)
    target_include_directories(shape_engine_bench_test PRIVATE include src
        ${OpenCV_INCLUDE_DIRS} vendor/shape_based_matching
        vendor/shape_based_matching/MIPP)
    target_link_libraries(shape_engine_bench_test PRIVATE ${OpenCV_LIBS}
        Threads::Threads)
    if(MSVC)
        target_compile_options(shape_engine_bench_test PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
    endif()
    if(VF_SHAPE_CUDA)
        target_sources(shape_engine_bench_test PRIVATE src/gpu/gpu_device.cu
            src/gpu/coarse_kernels.cu src/gpu/fine_windows.cu src/gpu/gpu_backend.cpp
            src/gpu/coarse_extract.cpp src/gpu/coarse_service.cpp src/gpu/fine_service.cpp)
        # PROFILE_STAGES 仅此目标开启：windows_ms 毫秒字段依赖该宏填充（窗口段
        # 门计量口径），trace JSON 输出经 VF_SHAPE_TRACE_QUIET 静默
        target_compile_definitions(shape_engine_bench_test PRIVATE VF_SHAPE_CUDA=1
            VF_SHAPE_PROFILE_STAGES=1)
    endif()
    add_test(NAME shape_engine_bench COMMAND shape_engine_bench_test)
    list(APPEND VF_TEST_NAMES shape_engine_bench)
    # W4 回退兜底门（真装配形态同 engine_bitexact）：三分支探测注入/初始化失败
    # 回退逐位/运行中失败 3 次粘性/auto 默认真跑 GPU + env=cpu 旁路。OFF 构建
    # 跳行 main 返回 77（ctest Skipped 防假绿，同 bitexact 口径）。
    add_executable(shape_gpu_fallback_test tests/gpu_fallback_test.cpp
        src/two_stage.cpp src/two_stage_train.cpp src/matcher.cpp src/polygon.cpp
        src/coarse_image.cpp src/rescue_score.cpp src/geometric_refinement.cpp
        vendor/shape_based_matching/line2Dup.cpp)
    target_include_directories(shape_gpu_fallback_test PRIVATE include src
        ${OpenCV_INCLUDE_DIRS} vendor/shape_based_matching
        vendor/shape_based_matching/MIPP)
    target_link_libraries(shape_gpu_fallback_test PRIVATE ${OpenCV_LIBS}
        Threads::Threads)
    if(MSVC)
        target_compile_options(shape_gpu_fallback_test PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
    endif()
    if(VF_SHAPE_CUDA)
        target_sources(shape_gpu_fallback_test PRIVATE src/gpu/gpu_device.cu
            src/gpu/coarse_kernels.cu src/gpu/fine_windows.cu src/gpu/gpu_backend.cpp
            src/gpu/coarse_extract.cpp src/gpu/coarse_service.cpp src/gpu/fine_service.cpp)
        target_compile_definitions(shape_gpu_fallback_test PRIVATE VF_SHAPE_CUDA=1)
    endif()
    add_test(NAME shape_gpu_fallback COMMAND shape_gpu_fallback_test)
    list(APPEND VF_TEST_NAMES shape_gpu_fallback)
    set_tests_properties(${VF_TEST_NAMES} PROPERTIES ENVIRONMENT "${VF_TEST_ENV}")
    set_tests_properties(shape_gpu_bitexact shape_engine_bitexact shape_engine_bench
        shape_gpu_fallback
        PROPERTIES SKIP_RETURN_CODE 77)
endif()
