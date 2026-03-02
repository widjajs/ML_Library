# numc — Matrix Library Reference

A minimal matrix library for neural network development in C, built on top of an arena allocator.
All matrix data is arena-allocated. There are no individual free functions — lifetime is managed
entirely by the arena.

---

## Matrix Struct

The core `Matrix` type stores a pointer to arena-allocated float data alongside its dimensions.
All functions operate on this struct passed by value. The `data` pointer always points into an
arena — never call `free()` on it directly.

---

## Allocation

### `init_mat`
Allocates a matrix of the given dimensions on the provided arena. If `zeroed` is true, all
elements are initialized to 0.0. Otherwise the memory is uninitialized. The returned `Matrix`
struct is passed by value; only the internal `data` array lives in the arena.

---

## Indexing

### `MAT_AT(mat, row, col)`
Macro that returns a `float *` pointer to the element at `[row][col]` using row-major indexing.
Can be used for both reads and writes via dereference.

---

## Initialization

### `mat_fill`
Sets every element of the matrix to the given constant value.

### `mat_fill_rand`
Fills the matrix with random values drawn from a normal distribution, scaled by the given factor.
Intended for weight initialization (Xavier / He init).

---

## Arithmetic

All arithmetic functions write their result into a pre-allocated `dest` matrix. The caller is
responsible for ensuring `dest` has the correct dimensions. This avoids hidden arena allocations
inside hot paths like the training loop.

### `mat_add`
Elementwise addition of `a` and `b`, written into `dest`. All three matrices must have identical
dimensions.

### `mat_sub`
Elementwise subtraction (`a - b`), written into `dest`. All three matrices must have identical
dimensions.

### `mat_scale`
Multiplies every element of `a` by the scalar value, written into `dest`.

### `mat_add_vec`
Broadcasts a bias row-vector across every row of `a` and writes the result into `dest`. Used to
add a bias term after a linear layer.

### `mat_mul`
Matrix multiplication of `a` and `b` (`a.cols` must equal `b.rows`), written into `dest`.
`dest` must be pre-allocated with dimensions `a.rows × b.cols`.

---

## Shape Operations

### `mat_transpose`
Returns a new matrix that is the transpose of `a`. Allocates the result on the provided arena.
The original matrix is unmodified.

### `mat_copy`
Returns a deep copy of `a` allocated on the provided arena. The new matrix has its own
independent `data` buffer.

### `mat_row`
Returns a `1 × cols` view into row `r` of `a`. No allocation is performed — the returned
matrix's `data` pointer points directly into the original matrix's memory. Do not use after
the source matrix's arena has been cleared.

---

## Activation Functions

All activation functions write elementwise results into a pre-allocated `dest` matrix.

### `mat_relu`
Applies ReLU elementwise: `max(0, x)` for each element of `a`, written into `dest`.

### `mat_relu_backward`
Computes the ReLU gradient: `gradient * (forward > 0)` elementwise, written into `dest`.
`forward` is the pre-activation values from the forward pass.

### `mat_softmax`
Applies softmax row-wise across `a`, written into `dest`. Used on the output layer to produce
a probability distribution over classes.

---

## Loss

### `mat_cross_entropy`
Computes the mean cross-entropy loss over a batch. Takes a matrix of predicted probabilities
(one row per sample), a label array of true class indices, and the number of samples. Returns
a single scalar loss value.

### `mat_softmax_grad`
Computes the combined gradient of the softmax + cross-entropy loss in a single pass, written
into `dest`. This is significantly simpler and more numerically stable than computing the two
gradients separately. `probs` is the softmax output from the forward pass.

---

## Debug

### `mat_print`
Prints the full contents of the matrix to stdout with a label. Intended for debugging small
matrices — do not use on large matrices during training.

### `mat_shape`
Prints just the matrix name and its dimensions in the format `name: (rows x cols)`. Useful for
quickly verifying shapes during development.

