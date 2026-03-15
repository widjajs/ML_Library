#ifndef MLLIB_H
#define MLLIB_H

#include "numc.h"
#include "utility.h"

typedef struct {
    u64 in_features;
    u64 hidden_size;
    u64 out_features;
    f64 lr;
    u64 batch_size;
    u64 epochs;
} NNConfig;

typedef struct {
    Matrix *W; // weights
    Matrix *b; // bias
} NNLayer;

typedef struct {
    NNLayer *hidden_1;
    NNLayer *output_layer;
} NNModel;

typedef struct {
    Matrix *Z1, *A1; // pre/post ReLU, respectively
    Matrix *Z2, *A2; // pre/post softmax, respectively
} NNCache;

typedef struct {
    Matrix *dw1, *db1;
    Matrix *dw2, *db2;
} NNGrads;

// init functions
NNConfig *NN_init_config(Arena *arena, u64 hidden_size, u64 batch_size, u64 epochs,
                         f64 lr, u64 in_features, u64 out_features);
NNModel *NN_init_model(Arena *arena, NNConfig *config);
NNCache *NN_init_cache(Arena *arena, NNConfig *config);
NNGrads *NN_init_grad(Arena *arena, NNConfig *cofig);

// training
void NN_forward(NNModel *model, Matrix *input, Matrix *output, NNCache *cache);
void NN_backward(NNModel *model, Matrix *input, Matrix *target, NNCache *cache);
void NN_update(NNModel *model, NNGrads *grads, f64 lr);
f64 NN_train_step(NNModel *model, Matrix *batch_x, Matrix *batch_y, NNCache *cache,
                  NNGrads *grads, f64 lr);

// utility
f64 NN_cross_entropy(Matrix *predictions, Matrix *targets);
f64 NN_accuracy(Matrix *predictions, Matrix *targets);

#endif
