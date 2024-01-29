#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "pthreads_barrier_mac.h"
#include "pthread.h"

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

ppm_image *image;
int step_x;
int step_y;
unsigned char **grid;
ppm_image **contour_map;
ppm_image *scaled_image;
int P; // number of threads
int p;
int q;
unsigned char sigma;
pthread_barrier_t barrier;
char *out_file;

// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map() {
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "../checker/contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++) {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}

void* thread_function(void* arg) {
    int thread_id = *(int*)arg;

    int scalingNecessary = 0;

    // 1. Rescale image (if necessary)
    uint8_t sample[3];
    if (image->x <= RESCALE_X && image->y <= RESCALE_Y) {
        scaled_image = image;
        scalingNecessary = 1;
    }
    if (scalingNecessary == 0) {
        // use bicubic interpolation for scaling
        int start = thread_id * scaled_image->x / P;
        int end = fmin((thread_id + 1) * (double)scaled_image->x / P, scaled_image->x);
        for (int i = start; i < end; i++) {
            for (int j = 0; j < scaled_image->y; j++) {
                float u = (float)i / (float)(scaled_image->x - 1);
                float v = (float)j / (float)(scaled_image->y - 1);
                sample_bicubic(image, u, v, sample);

                scaled_image->data[i * scaled_image->y + j].red = sample[0];
                scaled_image->data[i * scaled_image->y + j].green = sample[1];
                scaled_image->data[i * scaled_image->y + j].blue = sample[2];
            }
        }

        pthread_barrier_wait(&barrier);
    }

    // 2. Sample the grid
    p = scaled_image->x / step_x;
    q = scaled_image->y / step_y;

    pthread_barrier_wait(&barrier);

    int start = thread_id * p / P;
    int end = fmin((thread_id + 1) * (double)p / P, p);
    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = scaled_image->data[i * step_x * scaled_image->y + j * step_y];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > sigma) {
                grid[i][j] = 0;
            } else {
                grid[i][j] = 1;
            }
        }
    }
    grid[p][q] = 0;

    pthread_barrier_wait(&barrier);

    // last sample points have no neighbors below / to the right, so we use pixels on the
    // last row / column of the input image for them
    for (int i = start; i < end; i++) {
        ppm_pixel curr_pixel = scaled_image->data[i * step_x * scaled_image->y + scaled_image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[i][q] = 0;
        } else {
            grid[i][q] = 1;
        }
    }

    pthread_barrier_wait(&barrier);

    start = thread_id * q / P;
    end = fmin((thread_id + 1) * (double)q / P, q);
    for (int j = start; j < end; j++) {
        ppm_pixel curr_pixel = scaled_image->data[(scaled_image->x - 1) * scaled_image->y + j * step_y];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[p][j] = 0;
        } else {
            grid[p][j] = 1;
        }
    }

    pthread_barrier_wait(&barrier);

    // 3. March the squares
    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(scaled_image, contour_map[k], i * step_x, j * step_y);
        }
    }

    pthread_barrier_wait(&barrier);

    // 4. Write output
    if (thread_id == 0) {
        write_ppm(scaled_image, out_file);
    }

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }

    // read input
    image = read_ppm(argv[1]);
    P = atoi(argv[3]);
    step_x = STEP;
    step_y = STEP;
    p = image->x / step_x;
    q = image->y / step_y;
    sigma = SIGMA;
    contour_map = init_contour_map();
    out_file = argv[2];

    // alloc memory
    grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char*));
    if (!grid) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++) {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i]) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    scaled_image = (ppm_image *)malloc(sizeof(ppm_image));
    if (!scaled_image) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    scaled_image->x = RESCALE_X;
    scaled_image->y = RESCALE_Y;

    scaled_image->data = (ppm_pixel*)malloc(scaled_image->x * scaled_image->y * sizeof(ppm_pixel));
    if (!scaled_image) {
        fprintf(stderr, "Unable to allocate memoryHuya\n");
        exit(1);
    }

    pthread_t tid[P];
    int thread_id[P];
    pthread_barrier_init(&barrier, NULL, P);

    for (int i = 0; i < P; i++) {
        thread_id[i] = i;
        pthread_create(&(tid[i]), NULL, thread_function, &(thread_id[i]));
    }

    for (int i = 0; i < P; i++) {
        pthread_join(tid[i], NULL);
    }

    pthread_barrier_destroy(&barrier);

    // free resources
    free_resources(image, contour_map, grid, step_x);

    pthread_exit(NULL);
}