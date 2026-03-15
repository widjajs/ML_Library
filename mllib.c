#include "./include/mllib.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

NNConfig *NN_init_config(Arena *arena, u64 hidden_size, u64 batch_size, u64 epochs,
                         f64 lr, u64 in_features, u64 out_features) {
    NNConfig *config = arena_push(arena, sizeof(NNConfig), true);
    if (!config) return NULL;

    config->hidden_size = hidden_size;
    config->lr = lr;
    config->batch_size = batch_size;
    config->epochs = epochs;
    config->in_features = in_features;
    config->out_features = out_features;

    return config;
}

NNLayer *NN_init_layer(Arena *arena, u64 in_features, u64 out_features) {
    // in_features = # features coming from prev layer
    // out features = # features going to next layer
    NNLayer *layer = arena_push(arena, sizeof(NNLayer), true);
    if (!layer) return NULL;

    // ex: for MNIST 10 x 784
    layer->W = mat_init(arena, out_features, in_features, true);
    if (!layer->W) return NULL;

    // Xavier init: just set weights to random at the start
    f64 scale = sqrt(2.0 / (in_features + out_features));
    mat_fill_rand(layer->W, scale);

    layer->b = mat_init(arena, out_features, 1, true);
    if (!layer->b) return NULL;

    return layer;
}

NNModel *NN_init_model(Arena *arena, NNConfig *config) {
    NNModel *model = arena_push(arena, sizeof(NNModel), true);
    if (!model) return NULL;

    model->hidden_1 = NN_init_layer(arena, config->in_features, config->hidden_size);
    model->output_layer = NN_init_layer(arena, config->hidden_size, config->out_features);

    if (!model->hidden_1 || !model->output_layer) return NULL;

    return model;
}

NNCache *NN_init_cache(Arena *arena, NNConfig *config) {
    NNCache *cache = arena_push(arena, sizeof(NNCache), true);
    if (!cache) return NULL;

    cache->Z1 = mat_init(arena, config->hidden_size, config->batch_size, true);
    cache->A1 = mat_init(arena, config->hidden_size, config->batch_size, true);
    cache->Z2 = mat_init(arena, config->out_features, config->batch_size, true);
    cache->A2 = mat_init(arena, config->out_features, config->batch_size, true);

    return cache;
}

NNGrads *NN_init_grad(Arena *arena, NNConfig *config) {
    NNGrads *grads = arena_push(arena, sizeof(NNGrads), true);
    if (!grads) return NULL;

    grads->dw1 = mat_init(arena, config->hidden_size, config->in_features, true);
    grads->db1 = mat_init(arena, config->hidden_size, 1, true);
    grads->dw2 = mat_init(arena, config->out_features, config->hidden_size, true);
    grads->db2 = mat_init(arena, config->out_features, 1, true);

    return grads;
}

void NN_forward(NNModel *model, Matrix *input, Matrix *output, NNCache *cache);
void NN_backward(NNModel *model, Matrix *input, Matrix *target, NNCache *cache);
void NN_update(NNModel *model, NNGrads *grads, f64 lr);

f64 NN_train_step(NNModel *model, Matrix *batch_x, Matrix *batch_y, NNCache *cache,
                  NNGrads *grads, f64 lr);

f64 NN_cross_entropy(Matrix *predictions, Matrix *targets);
f64 NN_accuracy(Matrix *predictions, Matrix *targets);
