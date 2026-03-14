#define _GNU_SOURCE

#include <assert.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "./include/arena.h"
#include "./include/candas.h"
#include "./include/numc.h"
#include "./include/prng.h"

int main() {
    // u32 img_magic, img_count, img_rows, img_cols;

    // FILE *fp_images = fopen("./dataset/train-images.idx3-ubyte", "rb");
    Arena *arena = arena_init(GiB(1), sysconf(_SC_PAGESIZE));

    // read_image_headers(fp_images, &img_magic, &img_count, &img_rows, &img_cols);
    // printf("%d, %d, %d, %d\n", img_magic, img_count, img_rows, img_cols);

    // u64 total_pixels = img_count * img_rows * img_cols;
    // u8 *training_images = arena_push(arena, total_pixels, false);

    // read_images(fp_images, training_images, total_pixels);
    // draw_mnist_ascii(training_images + ((img_count - 215) * 28 * 28));
    // draw_mnist_color(training_images + ((img_count - 215) * 28 * 28));

    // u32 label_magic, labels_size;
    // FILE *fp_labels = fopen("./dataset/train-labels.idx1-ubyte", "rb");
    // read_label_headers(fp_labels, &label_magic, &labels_size);

    // printf("%d %d\n", label_magic, labels_size);

    // u8 *training_labels = arena_push(arena, labels_size, false);

    // read_labels(fp_labels, training_labels, labels_size);
    // printf("%d\n", training_labels[img_count - 40323]);

    // fclose(fp_images);
    // fclose(fp_labels);
    u8 labels5a[] = {0};
    u8 labels5b[] = {0, 0};
    Matrix *probs5a = mat_init(arena, 1, 3, true);
    Matrix *probs5b = mat_init(arena, 2, 3, true);
    Matrix *dest5a = mat_init(arena, 1, 3, true);
    Matrix *dest5b = mat_init(arena, 2, 3, true);

    *MAT_AT(probs5a, 0, 0) = 0.3;
    *MAT_AT(probs5a, 0, 1) = 0.5;
    *MAT_AT(probs5a, 0, 2) = 0.2;
    *MAT_AT(probs5b, 0, 0) = 0.3;
    *MAT_AT(probs5b, 0, 1) = 0.5;
    *MAT_AT(probs5b, 0, 2) = 0.2;
    *MAT_AT(probs5b, 1, 0) = 0.3;
    *MAT_AT(probs5b, 1, 1) = 0.5;
    *MAT_AT(probs5b, 1, 2) = 0.2;

    mat_softmax_grad(arena, dest5a, probs5a, labels5a);
    mat_softmax_grad(arena, dest5b, probs5b, labels5b);

    // batch of 2 should give exactly half the gradient of batch of 1
    assert(fabs(*MAT_AT(dest5a, 0, 0) - 2.0 * (*MAT_AT(dest5b, 0, 0))) < 1e-9);
    arena_delete(arena);

    return 0;
}
