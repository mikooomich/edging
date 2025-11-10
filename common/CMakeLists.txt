add_library(common
        bitmap.cpp
        utils.cpp
)


target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})