# NumC Matrix Library Specifications

Minimal arena-allocated matrix library for neural networks. All operations are in-place where possible. Assumes row-major storage with `MAT_AT(mat, row, col)` macro.

## Allocation & Initialization

### `Matrix *mat_init(Arena *arena, const u64 rows, const u64 cols, const bool zeroed)`

- Allocates `Matrix` struct + data array on arena
- `zeroed=true`: initializes data to 0.0
- Shape: `rows × cols`, returns `NULL` on failure

### `void mat_fill(Matrix *a, const f64 val)`

- Fills entire matrix with constant `val`
- In-place modification

### `void mat_fill_rand(Matrix *a, const f64 scale)`

- Fills with `randn() * scale` (Gaussian noise)
- Xavier/Glorot scaling support via `scale = sqrt(2/(fan_in + fan_out))`

## Arithmetic Operations

### `void mat_add(Matrix *dest, const Matrix *a, const Matrix *b)`

- `dest = a + b` (elementwise)
- Requires `a.shape == b.shape == dest.shape`
- Prints error and returns on shape mismatch

### `void mat_sub(Matrix *dest, const Matrix *a, const Matrix *b)`

- `dest = a - b` (elementwise)
- Shape validation identical to `mat_add`

### `void mat_mul(Matrix *dest, const Matrix *a, const Matrix *b)`

- Matrix multiplication: `dest[row,col] += a[row,i] * b[i,col]`
- Requires `dest.shape = (a.rows, b.cols)`, `a.cols == b.rows`
- Cache-friendly ijk loop order

### `void mat_mul_transpose(Matrix *dest, Matrix *a, Matrix *b, b32 a_transp, b32 b_transp)`

- General matmul with optional transposes: `dest = a^T · b` or `a · b^T` or `a^T · b^T`
- Computes dimensions dynamically, fills `dest` with zeros first
- Full generality for backprop (grad × activation^T)

### `void mat_scale(Matrix *dest, const Matrix *a, const f64 scalar)`

- `dest = a * scalar` (elementwise)
- In-place scalar multiplication

### `void mat_add_vec(Matrix *dest, const Matrix *a, const Matrix *bias)`

- `dest = a + bias` broadcasting `bias[1 × cols]` across all rows
- For layer bias addition: assumes `bias.shape = [1 × output_features]`

## Shape Operations

### `Matrix *mat_transpose(Arena *arena, const Matrix *a)`

- Allocates new `[cols × rows]` matrix with transposed data
- Returns `NULL` on allocation failure

### `Matrix *mat_copy(Arena *arena, const Matrix *a)`

- Deep copies matrix data to new arena allocation
- Identical shape

### `Matrix mat_row(const Matrix *a, u64 const row)`

- Returns row view: `{rows=1, cols=a.cols, data=&a.data[row*a.cols]}`
- **No allocation** — lightweight view for reductions

## Activation Functions

### `void mat_relu(Matrix *dest, const Matrix *a)`

- `dest = max(0, a)` elementwise
- In-place capable

### `void mat_relu_backward(Matrix *dest, const Matrix *grad, const Matrix *forward)`

- `dest = grad * (forward > 0)` (ReLU derivative)
- Requires `grad.shape == forward.shape`

### `void mat_softmax(Matrix *dest, const Matrix *a)`

- Row-wise softmax with numerical stability:
  1. Find row max
  2. `exp(x_i - max)`
  3. Normalize by sum
- Assumes `dest` same shape as `a`

## One-Hot & Loss Helpers

### `Matrix *mat_one_hot(Arena *arena, const u8 *labels, const u64 n, const u64 num_classes)`

- Creates `[n × num_classes]` one-hot matrix from label indices
- `one_hot[i, labels[i]] = 1.0`

### `f64 mat_cross_entropy(Arena *arena, const Matrix *probs, const u8 *labels)`

- `-mean(sum(one_hot × log(probs)))` over batch
- Uses temp arena for one-hot conversion
- **Monitoring only** (not for gradients)

### `void mat_softmax_grad(Arena *arena, Matrix *dest, const Matrix *probs, const u8 *labels)`

- Combined softmax+xentropy backward: `dest = (probs - one_hot) / batch_size`
- **Key optimization**: no explicit jacobian computation
- Temp arena for one-hot

## Debug & Inspection

### `void mat_print(Matrix *a, const char *name)`

- Pretty-prints matrix with shape preview
- Truncates large matrices (`>6×6`) with `...` ellipses
- Compact format: `%9.5f`

### `void mat_shape(Matrix *a)`

- Prints `(rows, cols)` tuple

## Key Design Decisions

- **Arena-only**: No manual freeing, tied to arena lifetime
- **In-place preferred**: Saves allocations during training loops
- **Shape validation**: Defensive error printing (no crashes)
- **Broadcasting**: `mat_add_vec` for efficient bias addition
- **Numerical stability**: Softmax max-subtraction, future log-sum-exp for loss
- **Backprop-ready**: Transpose mul, activation derivatives

## Usage Constraints

- All matrices row-major: `[batch × features]`
- No ownership transfer: caller manages arena lifetimes
- Error handling via stderr (extend with callbacks later)
- Single-threaded (add OpenMP pragmas for speed)

## Performance Notes

- Cache-friendly matmul (i outer, j middle, k inner)
- Minimal allocations (views where possible)
- No temporaries in core hot loops

# NN Library Function Specifications

## Config Functions

### `NNConfig *NN_init_config(Arena *arena, u64 hidden_size, u64 batch_size, u64 epochs, b32 has_hidden)`

- Allocates and initializes `NNConfig` struct on given arena
- Sets all fields: `hidden_size`, `lr` (default 0.01), `batch_size`, `epochs`, `has_hidden`
- Returns pointer to allocated config or `NULL` on failure
- Makes config immutable after creation

## Model Functions

### `NNModel *NN_init_model(Arena *arena, NNConfig *config)`

- Creates `NNModel` with **two layers** when `has_hidden=true`:
  - `layer_1`: hidden layer `[hidden_size × 784]` weights, `[hidden_size × 1]` bias (ReLU activation)
  - `layer_2`: output layer `[10 × hidden_size]` weights, `[10 × 1]` bias (Softmax)
- Single layer mode (`has_hidden=false`): `layer_1` as `[10 × 784]` direct to output
- Initializes weights randomly (Xavier/Glorot), biases to zero
- Returns allocated model or `NULL` on allocation failure

### `NNLayer *NN_init_layer(Arena *arena, u64 out_features, u64 in_features)`

- Allocates NNLayer with W[out_features × in_features], b[out_features × 1]
- Weights: Xavier init randn() * sqrt(2/(in_features + out_features))
- Biases: zeros

## Forward Pass

### `void NN_forward(NNModel *model, Matrix *input, Matrix *output, NNCache *cache)`

- **Two-layer forward** (`has_hidden=true`):
  1. `Z1 = layer_1.W · input + layer_1.b`
  2. `A1 = ReLU(Z1)` 
  3. `Z2 = layer_2.W · A1 + layer_2.b`
  4. `A2 = softmax(Z2)` → output
- **Single-layer** (`has_hidden=false`): `Z1 = layer_1.W · input + layer_1.b`, `A1 = softmax(Z1)`
- Stores all intermediates (`Z1,A1,Z2,A2`) in cache for backprop
- Input shape: `[784 × batch_size]`, Output: `[10 × batch_size]`
- Overwrites output matrix in-place

## Training Functions

### `void NN_backward(NNModel *model, Matrix *input, Matrix *target, NNCache *cache, NNGrads *grads)`

- **Two-layer backprop** (`has_hidden=true`):
  1. `dZ2 = A2 - target` (softmax+xentropy)
  2. `dW2 = (1/batch) · dZ2 · A1^T`, `db2 = mean(dZ2)`
  3. `dA1 = layer_2.W^T · dZ2`
  4. `dZ1 = dA1 ⊙ ReLU'(A1)`
  5. `dW1 = (1/batch) · dZ1 · input^T`, `db1 = mean(dZ1)`
- **Single-layer**: `dw1 = A1 - target`, `db1 = mean(dw1)`
- Stores all gradients (`dw1,db1,dw2,db2`) in grads struct

### `void NN_update(NNModel *model, NNGrads *grads, f64 lr)`

- Gradient descent: `W1 -= lr · dw1`, `b1 -= lr · db1`, same for layer 2
- In-place weight modification
- Optional gradient clipping (future)

## Cache & Grad Management

### `NNCache *NN_init_cache(Arena *arena, NNConfig *config)`

- Allocates cache matrices matching model:
  - Single layer: `Z1[10×batch], A1[10×batch]`
  - Two layers: `Z1[hidden×batch], A1[hidden×batch], Z2[10×batch], A2[10×batch]`
- Reused across forward/backward calls

### `NNGrads *NN_init_grads(Arena *arena, NNConfig *config)`

- Allocates gradient matrices matching model dimensions:
  - Single: `dw1[10×784], db1[10×1]`
  - Two layers: `dw1[hidden×784], db1[hidden×1], dw2[10×hidden], db2[10×1]`

## Training Loop Helper

### `f64 NN_train_step(NNModel *model, Matrix *batch_X, Matrix *batch_y, NNCache *cache, NNGrads *grads, f64 lr)`

- Single call: forward → backward → update
- Returns cross-entropy loss: `-mean(sum(target · log(preds)))`
- Validates matrix shapes, handles both single/two-layer modes

## Utility Functions

### `f64 NN_cross_entropy_loss(Matrix *predictions, Matrix *targets)`

- Average cross-entropy loss with log-sum-exp stability
- Expects softmax outputs, one-hot targets

### `f64 NN_accuracy(Matrix *predictions, Matrix *targets)`

- Classification accuracy: `mean(argmax(preds) == argmax(targets))`
- Batch-compatible
