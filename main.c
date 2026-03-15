#define _GNU_SOURCE

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "./include/arena.h"
#include "./include/candas.h"
#include "./include/numc.h"
#include "./include/prng.h"

void test_data_load_batch() {
    Arena *arena = arena_init(GiB(100), sysconf(_SC_PAGESIZE));

    // load MNIST data
    FILE *fp_images = fopen("./dataset/train-images.idx3-ubyte", "rb");
    FILE *fp_labels = fopen("./dataset/train-labels.idx1-ubyte", "rb");

    u32 img_magic, img_count, img_rows, img_cols;
    read_image_headers(fp_images, &img_magic, &img_count, &img_rows, &img_cols);

    u32 label_magic, labels_size;
    read_label_headers(fp_labels, &label_magic, &labels_size);

    u8 *pixels = arena_push(arena, img_count * 28 * 28, false);
    u8 *labels = arena_push(arena, labels_size, false);

    read_images(fp_images, pixels, img_count * 28 * 28);
    read_labels(fp_labels, labels, labels_size);

    fclose(fp_images);
    fclose(fp_labels);

    // ── Test 1: Single image (first image is digit 1) ───────────────────────
    u64 indices1[1] = {0};
    Matrix *x1 = mat_init(arena, 1, 784, true);
    u8 y1[1];

    data_load_batch(indices1, x1, y1, labels, pixels, 0, 1);

    assert(x1->rows == 1 && x1->cols == 784);
    assert(y1[0] == labels[0]); // first MNIST image is digit 1
    assert(*MAT_AT(x1, 0, 0) >= 0.0 && *MAT_AT(x1, 0, 0) <= 1.0);

    printf("Test 1 PASSED: single image\n");

    // ── Test 2: Verify exact pixel extraction + normalization ────────────────
    u64 indices2[2] = {0, 1}; // first two images
    Matrix *x2 = mat_init(arena, 2, 784, true);
    u8 y2[2];

    data_load_batch(indices2, x2, y2, labels, pixels, 0, 2);

    // verify labels
    assert(y2[0] == labels[0]);
    assert(y2[1] == labels[1]);

    // verify first pixel of first image normalized correctly
    u8 raw_pixel = pixels[0];
    f64 norm_pixel = *MAT_AT(x2, 0, 0);
    assert(fabs(norm_pixel - raw_pixel / 255.0) < 1e-9);

    // verify first pixel of second image
    raw_pixel = pixels[1 * 784];
    norm_pixel = *MAT_AT(x2, 1, 0);
    assert(fabs(norm_pixel - raw_pixel / 255.0) < 1e-9);

    for (u64 i = 0; i < 784; i++) {
        if (pixels[i] != 0) {
            assert(*MAT_AT(x2, 0, i) == pixels[i] / 255.0);
        }
    }

    printf("Test 2 PASSED: pixel extraction\n");

    // ── Test 3: Shuffled indices work ────────────────────────────────────────
    u64 indices3[3] = {0, 1, 2};
    data_shuffle(indices3, 3); // randomize order

    Matrix *x3 = mat_init(arena, 3, 784, true);
    u8 y3[3];

    data_load_batch(indices3, x3, y3, labels, pixels, 0, 3);

    // verify labels match shuffled indices
    for (u64 i = 0; i < 3; i++)
        assert(y3[i] == labels[indices3[i]]);

    printf("Test 3 PASSED: shuffled indices\n");

    // ── Test 4: Batch from middle of dataset ─────────────────────────────────
    u64 indices4[2] = {10000, 10001};
    Matrix *x4 = mat_init(arena, 2, 784, true);
    u8 y4[2];

    data_load_batch(indices4, x4, y4, labels, pixels, 0, 2);

    assert(y4[0] == labels[10000]);
    assert(y4[1] == labels[10001]);

    printf("Test 4 PASSED: middle dataset\n");

    // ── Test 5: Full batch size 32 ───────────────────────────────────────────
    u64 *indices5 = malloc(32 * sizeof(u64));
    for (u64 i = 0; i < 32; i++)
        indices5[i] = i * 1000; // spaced out
    data_shuffle(indices5, 32);

    Matrix *x5 = mat_init(arena, 32, 784, true);
    u8 y5[32];

    data_load_batch(indices5, x5, y5, labels, pixels, 0, 32);

    // verify shape and normalization
    assert(x5->rows == 32 && x5->cols == 784);

    // spot check labels and first pixel
    for (u64 i = 0; i < 32; i++) {
        assert(y5[i] == labels[indices5[i]]);
        assert(*MAT_AT(x5, i, 0) >= 0.0 && *MAT_AT(x5, i, 0) <= 1.0);
    }

    free(indices5);
    printf("Test 5 PASSED: full batch\n");

    // ── Test 6: End of dataset (no out-of-bounds) ────────────────────────────
    u64 indices6[3] = {img_count - 3, img_count - 2, img_count - 1};
    Matrix *x6 = mat_init(arena, 3, 784, true);
    u8 y6[3];

    data_load_batch(indices6, x6, y6, labels, pixels, 0, 3);

    for (u64 i = 0; i < 3; i++)
        assert(y6[i] == labels[img_count - 3 + i]);

    printf("Test 6 PASSED: end of dataset\n");

    arena_delete(arena);
    printf("All data_load_batch tests passed!\n");
}

int main() {
    test_data_load_batch();
    return 0;
}
