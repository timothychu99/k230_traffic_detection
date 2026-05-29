cmake_minimum_required(VERSION 3.2)
project(triple_cam_yolo_mosaic C CXX)

get_filename_component(K230_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(k230_sdk "${K230_SDK_ROOT}")
set(nncase_sdk_root "${K230_SDK_ROOT}/src/big/nncase")
set(YOLO_DETECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ai_poc/yolo_detect")
set(AI_POC_CMAKE "${CMAKE_CURRENT_SOURCE_DIR}/../ai_poc/cmake")

option(BUILD_TRIPLE_CAM_MOSAIC_LINUX_GLIBC
       "ON: build only triple_cam_mosaic_person_reader.elf using Riscv64_glibc.cmake (little-core Linux). OFF (default): build triple_cam_yolo_mosaic.elf with musl Riscv64.cmake (RTOS)."
       OFF)

if(BUILD_TRIPLE_CAM_MOSAIC_LINUX_GLIBC)
  if(CMAKE_CXX_COMPILER MATCHES "musl")
    message(FATAL_ERROR
      "BUILD_TRIPLE_CAM_MOSAIC_LINUX_GLIBC=ON is for the little-core Linux (glibc) toolchain "
      "(Riscv64_glibc.cmake). This compiler is musl: ${CMAKE_CXX_COMPILER}")
  endif()
else()
  if(CMAKE_CXX_COMPILER MATCHES "unknown-linux-gnu" AND NOT CMAKE_CXX_COMPILER MATCHES "musl")
    message(FATAL_ERROR
      "triple_cam_yolo_mosaic.elf must be built with the musl cross compiler "
      "(Riscv64.cmake, e.g. riscv64-unknown-linux-musl-g++). Current: ${CMAKE_CXX_COMPILER}. "
      "For triple_cam_mosaic_person_reader.elf use Riscv64_glibc.cmake and "
      "-DBUILD_TRIPLE_CAM_MOSAIC_LINUX_GLIBC=ON (see build_linux_fifo_reader.sh).")
  endif()
endif()

set(k230_opencv ${k230_sdk}/src/big/utils/lib/opencv)
include_directories(${k230_opencv}/include/opencv4/)
link_directories(${k230_opencv}/lib ${k230_opencv}/lib/opencv4/3rdparty)
link_directories(${k230_sdk}/src/big/mpp/userapps/lib)

include_directories(${nncase_sdk_root}/riscv64)
include_directories(${nncase_sdk_root}/riscv64/nncase/include)
include_directories(${nncase_sdk_root}/riscv64/nncase/include/nncase/runtime)
link_directories(${nncase_sdk_root}/riscv64/nncase/lib/)
include_directories(${YOLO_DETECT_DIR})
include_directories(${nncase_sdk_root}/riscv64/rvvlib/include)
include_directories(${k230_sdk}/src/big/mpp/userapps/api/)
include_directories(${k230_sdk}/src/big/mpp/include)
include_directories(${k230_sdk}/src/big/mpp/include/comm)
include_directories(${k230_sdk}/src/big/mpp/userapps/sample/sample_vo)
link_directories(${nncase_sdk_root}/riscv64/rvvlib/)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${K230_SDK_ROOT}/src/common/cdk/user/component/datafifo/include)
link_directories(${K230_SDK_ROOT}/src/common/cdk/user/component/datafifo/host/lib)

set(src
    main.cc
    person_fifo_writer.cpp
    ${YOLO_DETECT_DIR}/ai_base.cc
    ${YOLO_DETECT_DIR}/utils.cc
    ${YOLO_DETECT_DIR}/yolo_detect.cc
)
set(bin triple_cam_yolo_mosaic.elf)
set(fifo_reader_src main_linux.cpp)
set(fifo_reader_bin triple_cam_mosaic_person_reader.elf)
set(TRIPLE_CAM_LINK_SCRIPT "${AI_POC_CMAKE}/link.lds")

if(NOT BUILD_TRIPLE_CAM_MOSAIC_LINUX_GLIBC)
  add_executable(${bin} ${src})
  target_compile_options(${bin} PRIVATE -pthread)
  target_link_options(${bin} PRIVATE "-T${TRIPLE_CAM_LINK_SCRIPT}" "--static" -pthread)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${bin} OpenMP::OpenMP_CXX)
  endif()
  target_link_libraries(${bin} -Wl,--start-group rvv Nncase.Runtime.Native nncase.rt_modules.k230 functional_k230 sys vicap vb cam_device cam_engine
   hal oslayer ebase fpga isp_drv binder auto_ctrol common cam_caldb isi 3a buffer_management cameric_drv video_in virtual_hal start_engine cmd_buffer
   switch cameric_reg_drv t_database_c t_mxml_c t_json_c t_common_c vo connector sensor atomic dma
   ${K230_SDK_ROOT}/src/common/cdk/user/component/datafifo/host/lib/libdatafifo.a pthread -Wl,--end-group)
  target_link_libraries(${bin} opencv_imgcodecs opencv_imgproc opencv_core libjpeg-turbo libopenjp2 libpng libtiff libwebp csi_cv zlib)

  install(TARGETS ${bin} DESTINATION bin)
endif()

if(BUILD_TRIPLE_CAM_MOSAIC_LINUX_GLIBC)
  add_executable(${fifo_reader_bin} ${fifo_reader_src})
  target_compile_features(${fifo_reader_bin} PRIVATE cxx_std_17)
  set_target_properties(${fifo_reader_bin} PROPERTIES LINKER_LANGUAGE CXX)
  target_link_options(${fifo_reader_bin} PRIVATE -static)
  target_include_directories(${fifo_reader_bin} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${K230_SDK_ROOT}/src/big/mpp/include
    ${K230_SDK_ROOT}/src/common/cdk/user/component/datafifo/include
  )
  target_link_directories(${fifo_reader_bin} PRIVATE
    ${K230_SDK_ROOT}/src/big/mpp/userapps/lib
  )
  target_link_libraries(${fifo_reader_bin}
    -Wl,--start-group
    oslayer
    ebase
    ${K230_SDK_ROOT}/src/common/cdk/user/component/datafifo/slave/lib/libdatafifo.a
    sys
    pthread
    m
    dl
    atomic
    -Wl,--end-group
  )
  install(TARGETS ${fifo_reader_bin} DESTINATION bin)
endif()
