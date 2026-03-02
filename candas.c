#include "candas.h"
#include "arena.h"

#include <stdio.h>

u32 swap_endian(u32 x) {
    return ((x >> 24) & (0x000000FF)) | ((x >> 8) & (0x0000FF00)) | ((x << 8) & (0x00FF0000)) |
           ((x << 24) & (0xFF000000));
}

int read_label_headers(FILE *fp, u32 *magic, u32 *size) {
    if (!fp) return FILE_ERR;

    fread(magic, 4, 1, fp);
    fread(size, 4, 1, fp);

    // input file originally in Big-Endian
    *magic = swap_endian(*magic);
    *size = swap_endian(*size);

    return SUCCESS;
} /* read_label_headers() */

int read_image_headers(FILE *fp, u32 *magic, u32 *count, u32 *rows, u32 *cols) {
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
}

int read_images(FILE *fp, u8 *images, u64 size) {
    if (!fp) return FILE_ERR;
    if (fread(images, size, 1, fp) != 1) {
        return READ_ERR;
    }
    return SUCCESS;
} /* read_images() */

void draw_ascii_image(u8 *pixels, int idx, int rows, int cols) {
    // don't ask how this works
    u8 *img = pixels + (idx * cols * rows);
    char *chars = " .:-=+*#%@"; // darker pixels = denser characters

    printf("Image Index: %d\n", idx);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            u8 pixel = img[r * 28 + c];
            printf("%c%c", chars[pixel / 26], chars[pixel / 26]);
        }
        printf("\n");
    }
} /* draw_ascii_image() */

void draw_mnist_digit(u8 *image) {
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
} /* draw_ascii_image() */

int main() {
    u32 magic, count, rows, cols;

    FILE *fp = fopen("./dataset/train-images.idx3-ubyte", "rb");
    Arena *arena = arena_init(GiB(1), sysconf(_SC_PAGESIZE));

    read_image_headers(fp, &magic, &count, &rows, &cols);
    printf("%d, %d, %d, %d\n", magic, count, rows, cols);

    u64 total_pixels = count * rows * cols;
    u8 *training_images = arena_push(arena, total_pixels, false);

    read_images(fp, training_images, total_pixels);
    draw_mnist_digit(training_images + ((count - 1) * 28 * 28));

    fclose(fp);
    arena_delete(arena);

    return 0;
}

// TODO: read labels, put training image data into matrices, start working on numc
