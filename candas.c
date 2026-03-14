#include "./include/candas.h"

u32 swap_endian(u32 x) {
    return ((x >> 24) & (0x000000FF)) | ((x >> 8) & (0x0000FF00)) |
           ((x << 8) & (0x00FF0000)) | ((x << 24) & (0xFF000000));
} /* swap_endian() */

u32 read_label_headers(FILE *fp, u32 *magic, u32 *size) {
    if (!fp) return FILE_ERR;

    fread(magic, 4, 1, fp);
    fread(size, 4, 1, fp);

    // input file originally in Big-Endian
    *magic = swap_endian(*magic);
    *size = swap_endian(*size);

    return SUCCESS;
} /* read_label_headers() */

u32 read_labels(FILE *fp, u8 *labels, u64 size) {
    if (!fp) return FILE_ERR;
    if (fread(labels, size, 1, fp) != 1) {
        return READ_ERR;
    }
    return SUCCESS;

} /* read_labels() */

u32 read_image_headers(FILE *fp, u32 *magic, u32 *count, u32 *rows, u32 *cols) {
    if (!fp) return FILE_ERR;

    // read in header data
    fread(magic, 4, 1, fp);
    fread(count, 4, 1, fp);
    fread(rows, 4, 1, fp);
    fread(cols, 4, 1, fp);

    // input file originally in Big-Endian
    *magic = swap_endian(*magic);
    *count = swap_endian(*count);
    *rows = swap_endian(*rows);
    *cols = swap_endian(*cols);

    return SUCCESS;
} /* read_label_headers() */

u32 read_images(FILE *fp, u8 *images, u64 size) {
    if (!fp) return FILE_ERR;
    if (fread(images, size, 1, fp) != 1) {
        return READ_ERR;
    }
    return SUCCESS;
} /* read_images() */

/* Fisher-Yates shuffling algorithm */
void data_shuffle(u64 *indices, u64 n) {
    u64 k, temp;
    for (u64 i = n - 1; i > 0; i--) {
        // rand num [0, n]
        k = next() % (i + 1);

        temp = indices[i];
        indices[i] = indices[k];
        indices[k] = temp;
    }
}

void data_load_batch(u64 *indices, Matrix *x_out, u8 *y_out, const u8 *labels,
                     const u8 *pixels, const u64 start, const u64 batch_size) {
    for (u64 i = 0; i < batch_size; i++) {
        u64 idx = indices[i] * 784;
        for (u64 j = 0; j < 784; j++) {
            // copy image pixels
            *MAT_AT(x_out, i, j) = pixels[idx + j] / 255.0;
        }
    }

    for (u64 i = 0; i < batch_size; i++) {
        // copy answers
        y_out[i] = labels[indices[i + start]];
    }
}

void draw_mnist_ascii(u8 *image) {
    // don't ask how this works
    char *chars = " .:-=+*#%@"; // darker pixels = denser characters

    for (u32 r = 0; r < 28; r++) {
        for (u32 c = 0; c < 28; c++) {
            u8 pixel = image[r * 28 + c];
            printf("%c%c", chars[pixel / 26], chars[pixel / 26]);
        }
        printf("\n");
    }
} /* draw_mnist_ascii() */

void draw_mnist_color(u8 *image) {
    // don't ask how this works
    for (u32 r = 0; r < 28; r++) {
        for (u32 c = 0; c < 28; c++) {
            f32 num = image[r * 28 + c];
            u32 col = 232 + (u32)(num * 23);
            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\n");
    }
    printf("\x1b[0m");
} /* draw_mnist_color() */
