cmake_minimum_required(VERSION 3.14)

if("${BUILD_TARGET}" STREQUAL "server")
    add_subdirectory(server)
endif()

if("${BUILD_TARGET}" STREQUAL "client")
    add_subdirectory(client)
endif()
