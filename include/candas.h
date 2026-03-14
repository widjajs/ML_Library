#ifndef CANDAS_H
#define CANDAS_H

#include "numc.h"
#include "prng.h"
#include "utility.h"

#define FILE_ERR (-1)
#define READ_ERR (-2)
#define SUCCESS (1)

u32 swap_endian(u32 x);

u32 read_label_headers(FILE *fp, u32 *magic, u32 *size);
u32 read_image_headers(FILE *fp, u32 *magic, u32 *count, u32 *rows, u32 *cols);
u32 read_images(FILE *fp, u8 *images, u64 size);
u32 read_labels(FILE *fp, u8 *labels, u64 size);

void data_shuffle(u64 *indices, u64 n);
void data_load_batch(u64 *indices, Matrix *x_out, u8 *y_out, const u8 *labels,
                     const u8 *pixels, const u64 start, const u64 batch_size);

void draw_mnist_ascii(u8 *image);
void draw_mnist_color(u8 *image);

#endif
