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
// TODO: put training image data into matrices, start working on numc
