#include "./include/mllib.h"

NNConfig *NN_init_config(Arena *arena, u64 hidden_size, u64 batch_size, u64 epochs);
NNModel *NN_init_model(Arena *arena, NNConfig *config);
NNCache *NN_init_cache(Arena *arena, NNConfig *config);
NNGrads *NN_init_grad(Arena *arena, NNConfig *cofig);

void NN_forward(NNModel *model, Matrix *input, Matrix *output, NNCache *cache);
void NN_backward(NNModel *model, Matrix *input, Matrix *target, NNCache *cache);
