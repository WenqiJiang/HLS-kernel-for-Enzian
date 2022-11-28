#pragma once

#include <ap_int.h>
#include <hls_stream.h>

#include "constants.hpp"

typedef struct {
    int obj_id;
    // minimum bounding rectangle 
    float left; 
    float right; 
    float top; 
    float bottom; 
} obj_t; 

typedef struct {
    int obj_id_A;
    int obj_id_B;
} result_t;
