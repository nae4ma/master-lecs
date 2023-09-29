/*　3807024K 鍋島虎太郎
 * 
 *  Sobelフィルタによる，勾配方向の可視化
 *  1枚のPGM画像を読み込み，水平方向・垂直方向の画像を出力
 *
 * 　(コンパイル・実行例)
 *   gcc 3807024k_sobel.c
 *   .\a.exe sample.pgm result_h.ppm result_v.ppm
 *
 *   (実行環境)
 *   gcc version 12.2.0
 *   windows 11
 */


#include <stdio.h>
#include <stdlib.h>

#define HEIGHT 256
#define WIDTH  256

#define IPR_GRAD_H 0    //水平方向: 0
#define IPR_GRAD_V 1    //垂直方向: 1

void ipr_load_pgm(unsigned char image[][WIDTH], const char path[]);
void ipr_save_ppm(unsigned char image[][WIDTH][3], const char path[]);

void ipr_sobel(double **sobel, unsigned char image[][WIDTH], int direction);

int main(int argc, char *argv[])
{
    unsigned char src_image[HEIGHT][WIDTH];
    unsigned char dst_image[HEIGHT][WIDTH][3];

    double **sobel, **sobelX;
    sobel = (double**)malloc(sizeof(double*)*HEIGHT);
    sobelX = (double**)malloc(sizeof(double*)*HEIGHT);

    for(int i = 0; i < HEIGHT; i++){
        sobel[i] = (double*)malloc(sizeof(double)*WIDTH);
        sobelX[i] = (double*)malloc(sizeof(double)*WIDTH);
    }

    ipr_load_pgm(src_image, argv[1]);

    //水平方向の計算
    ipr_sobel(sobelX, src_image, 0);

    for (int m = 0; m < HEIGHT; m++) {
        for (int n = 0; n < WIDTH; n++) {

            // 基本の背景は白
            dst_image[m][n][2] = (unsigned char) 255;
            dst_image[m][n][0] = (unsigned char) 255;
            dst_image[m][n][1] = (unsigned char) 255;

            // 勾配が正
            if(sobelX[m][n] > 0.0){
                dst_image[m][n][0] = (unsigned char) 255;
                dst_image[m][n][2] = 255 - (unsigned char) sobelX[m][n];
                dst_image[m][n][1] = 255 - (unsigned char) sobelX[m][n];
            }
            // 勾配が負
            else if(sobelX[m][n] < 0.0){
                dst_image[m][n][2] = (unsigned char) 255;
                dst_image[m][n][0] = 255 - (unsigned char) sobelX[m][n] * -1.0;
                dst_image[m][n][1] = 255 - (unsigned char) sobelX[m][n] * -1.0;
            }

        }
    }
    
    // 水平方向の保存
    ipr_save_ppm(dst_image, argv[2]);

    //垂直方向の計算
    ipr_sobel(sobelX, src_image, 1);

        for (int m = 0; m < HEIGHT; m++) {
        for (int n = 0; n < WIDTH; n++) {

            // 基本の背景は白
            dst_image[m][n][2] = (unsigned char) 255;
            dst_image[m][n][0] = (unsigned char) 255;
            dst_image[m][n][1] = (unsigned char) 255;

            // 勾配が正
            if(sobelX[m][n] > 0.0){
                dst_image[m][n][0] = (unsigned char) 255;
                dst_image[m][n][2] = 255 - (unsigned char) sobelX[m][n];
                dst_image[m][n][1] = 255 - (unsigned char) sobelX[m][n];
            }
            // 勾配が負
            else if(sobelX[m][n] < 0.0){
                dst_image[m][n][2] = (unsigned char) 255;
                dst_image[m][n][0] = 255 - (unsigned char) sobelX[m][n] * -1.0;
                dst_image[m][n][1] = 255 - (unsigned char) sobelX[m][n] * -1.0;
            }

        }
    }

    // 垂直方向の保存
    ipr_save_ppm(dst_image, argv[3]);

    // メモリ解放
    for(int i = 0; i < HEIGHT; i++){
        free(sobel[i]);
        free(sobelX[i]);
    }

    return 0;
}

void ipr_load_pgm(unsigned char image[][WIDTH], const char path[])
{
    char magic_number[2];
    int width, height;
    int max_intensity;
    FILE *fp;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "%s が開けませんでした．\n", path);
        exit(1);
    }

    fscanf(fp, "%c%c", &magic_number[0], &magic_number[1]);
    if (magic_number[0] != 'P' || magic_number[1] != '5') {
        fprintf(stderr, "%s はバイナリ型 PGM ではありません．\n", path);
        fclose(fp);
        exit(1);
    }

    fscanf(fp, "%d %d", &width, &height);
    if (width != WIDTH || height != HEIGHT) {
        fprintf(stderr, "画像のサイズが異なります．\n");
        fprintf(stderr, "  想定サイズ：WIDTH = %d, HEIGHT = %d\n", WIDTH, HEIGHT);
        fprintf(stderr, "  実サイズ：  width = %d, height = %d\n", width, height);
        fclose(fp);
        exit(1);
    }

    fscanf(fp, "%d", &max_intensity);
    if (max_intensity != 255) {
        fprintf(stderr, "最大階調値が不正な値です（%d）．\n", max_intensity);
        fclose(fp);
        exit(1);
    }

    fgetc(fp);  // 最大階調値の直後の改行コードを読み捨て

    fread(image, sizeof(unsigned char), HEIGHT * WIDTH, fp);

    fclose(fp);
}

void ipr_save_ppm(unsigned char image[][WIDTH][3], const char path[])
{
    FILE *fp;
    
    fp = fopen(path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "%s が開けませんでした．\n", path);
        exit(1);
    }
    
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", WIDTH, HEIGHT);
    fprintf(fp, "255\n");
    fwrite(image, sizeof(unsigned char), HEIGHT * WIDTH * 3, fp);
    
    fclose(fp);
}

// ソーベルフィルタの計算
void ipr_sobel(double **sobel, unsigned char image[][WIDTH], int direction){

    if(direction == IPR_GRAD_H){                      //水平方向

        // (0, 0) の計算
        sobel[0][0]
            = image[0][1] + 2 * image[0][1] + image[1][1]
            - image[0][0] - 2 * image[0][0] - image[1][0];

        // (0, W - 1) の計算
        sobel[0][WIDTH - 1]
            = image[0][WIDTH - 1] + 2 * image[0][WIDTH - 1] + image[1][WIDTH - 1]
            - image[0][WIDTH - 2] - 2 * image[0][WIDTH - 2] - image[1][WIDTH - 2];

        // (H - 1, 0) の計算
        sobel[HEIGHT - 1][0]
            = image[HEIGHT - 2][1] + 2 * image[HEIGHT - 1][1] + image[HEIGHT - 1][1]
            - image[HEIGHT - 2][0] - 2 * image[HEIGHT - 1][0] - image[HEIGHT - 1][0];

        // (H - 1, W - 1) の計算
        sobel[HEIGHT - 1][WIDTH - 1]
            = image[HEIGHT - 2][WIDTH - 1] + 2 * image[HEIGHT - 1][WIDTH - 1]
            + image[HEIGHT - 1][WIDTH - 1]
            - image[HEIGHT - 2][WIDTH - 2] - 2 * image[HEIGHT - 1][WIDTH - 2]
            - image[HEIGHT - 1][WIDTH - 2];


        // 上端の (0,　1) -- (0,　W - 2) と
        // 下端の (H - 1, 1) -- (H - 1, W - 2) の計算
        for (int x = 1; x < WIDTH - 2; x++) {
            sobel[0][x]
                = image[0][x + 1] + 2 * image[0][x + 1] + image[1][x + 1]
                - image[0][x - 1] - 2 * image[0][x - 1] - image[1][x - 1];

            sobel[HEIGHT - 1][x]
                = image[HEIGHT - 2][x + 1] + 2 * image[HEIGHT - 1][x + 1] + image[HEIGHT - 1][x + 1]
                - image[HEIGHT - 2][x - 1] - 2 * image[HEIGHT - 1][x - 1] - image[HEIGHT - 1][x - 1];
        }

        // 左端である (1, 0)     -- (H - 2, 0)     と
        // 右端である (1, W - 1) -- (H - 2, W - 1) の計算
        for (int y = 1; y < HEIGHT - 2; y++) {
            sobel[y][0]
                = image[y - 1][1] + 2 * image[y][1] + image[y + 1][1]
                - image[y - 1][0] - 2 * image[y][0] - image[y + 1][0];

            sobel[y][WIDTH - 1]
                = image[y - 1][WIDTH - 1] + 2 * image[y][WIDTH - 1] + image[y + 1][WIDTH - 1]
                - image[y - 1][WIDTH - 2] - 2 * image[y][WIDTH - 2] - image[y + 1][WIDTH - 2];
        }

        // 端を除く領域の計算
        for (int y = 1; y < HEIGHT - 1; y++) {
            for (int x = 1; x < WIDTH - 1; x++) {
                sobel[y][x]
                    = image[y - 1][x + 1] + 2 * image[y][x + 1] + image[y + 1][x + 1]
                    - image[y - 1][x - 1] - 2 * image[y][x - 1] - image[y + 1][x - 1];
            }
        }
    }

    if(direction == IPR_GRAD_V){              //垂直方向

        // (0, 0) の計算
        sobel[0][0]
            = image[1][0] + 2 * image[1][0] + image[1][1]
            - image[0][0] - 2 * image[0][0] - image[0][1];

        // (0, W - 1) の計算
        sobel[0][WIDTH - 1]
            = image[1][WIDTH - 2] + 2 * image[1][WIDTH - 1] + image[1][WIDTH - 1]
            - image[0][WIDTH - 2] - 2 * image[0][WIDTH - 1] - image[0][WIDTH - 1];

        // (H - 1, 0) の計算
        sobel[HEIGHT - 1][0]
            = image[HEIGHT - 1][0] + 2 * image[HEIGHT - 1][0] + image[HEIGHT - 1][1]
            - image[HEIGHT - 2][0] - 2 * image[HEIGHT - 2][0] - image[HEIGHT - 2][1];

        // (H - 1, W - 1) の計算
        sobel[HEIGHT - 1][WIDTH - 1]
            = image[HEIGHT - 1][WIDTH - 2] + 2 * image[HEIGHT - 1][WIDTH - 1]
            + image[HEIGHT - 1][WIDTH - 1]
            - image[HEIGHT - 2][WIDTH - 2] - 2 * image[HEIGHT - 2][WIDTH - 1]
            - image[HEIGHT - 2][WIDTH - 1];


        // 上端である (0, 1) -- (0, W - 2) と
        // 下端である (H - 1, 1) -- (H - 1, W - 2) の計算
        for (int x = 1; x < WIDTH - 2; x++) {
            sobel[0][x]
                = image[1][x - 1] + 2 * image[1][x] + image[1][x + 1]
                - image[0][x - 1] - 2 * image[0][x] - image[0][x + 1];

            sobel[HEIGHT - 1][x]
                = image[HEIGHT - 1][x - 1] + 2 * image[HEIGHT - 1][x] + image[HEIGHT - 1][x + 1]
                - image[HEIGHT - 2][x - 1] - 2 * image[HEIGHT - 2][x] - image[HEIGHT - 2][x + 1];
        }

        // 左端である (1, 0)     -- (H - 2, 0)     と
        // 右端である (1, W - 1) -- (H - 2, W - 1) の計算
        for (int y = 1; y < HEIGHT - 2; y++) {
            sobel[y][0]
                = image[y + 1][0] + 2 * image[y + 1][0] + image[y + 1][1]
                - image[y - 1][0] - 2 * image[y - 1][0] - image[y - 1][1];

            sobel[y][WIDTH - 1]
                = image[y + 1][WIDTH - 2] + 2 * image[y + 1][WIDTH - 1] + image[y + 1][WIDTH - 1]
                - image[y - 1][WIDTH - 2] - 2 * image[y - 1][WIDTH - 1] - image[y - 1][WIDTH - 1];
        }

        // 端を除く領域の計算
        for (int y = 1; y < HEIGHT - 1; y++) {
            for (int x = 1; x < WIDTH - 1; x++) {
                sobel[y][x]
                    = image[y + 1][x - 1] + 2 * image[y + 1][x] + image[y + 1][x + 1]
                    - image[y - 1][x - 1] - 2 * image[y - 1][x] - image[y - 1][x + 1];
            }
        }
    }

    // 正規化処理
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            sobel[y][x] *= 0.125;
        }
    }
}
