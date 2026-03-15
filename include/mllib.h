#ifndef MLLIB_H
#define MLLIB_H

#include "numc.h"
#include "utility.h"

typedef struct {
    u64 hidden_size;
    f64 lr;
    u64 batch_size;
    u64 epochs;
    b32 has_hidden;
} NNConfig;

typedef struct {
    Matrix *W; // weights
    Matrix *b; // bias
} NNLayer;

typedef struct {
    NNLayer *layer_1;
    NNLayer *layer_2;
} NNModel;

typedef struct {
    Matrix *Z1, *A1; // pre/post ReLU, respectively
    Matrix *Z2, *A2; // pre/post softmax, respectively
} NNCache;

typedef struct {
    Matrix *dw1, *db1;
    Matrix *dw2, *db2;
} NNGrads;

#endif
