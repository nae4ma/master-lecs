 /*　3807024K 鍋島虎太郎
 * 
 *  Mean-Shift法による追跡
 *  100枚のPPM画像を読み込み，追跡結果を保存する
 *  モデルとの類似度が最大となる位置から少しずれた位置を初期位置とする
 *
 * 　(コンパイル・実行例)
 *   あらかじめ，src_imageフォルダ+sample[00000-00099].ppm，result_imageフォルダを用意
 *   gcc 3807024k_meanShift.c
 *   .\a.exe model_image.ppm
 *
 *   (実行環境)
 *   gcc version 12.2.0
 *   windows 11
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH           640     // 画像の横画素数
#define HEIGHT          480     // 画像の縦画素数
#define MODEL_WIDTH     69      // モデルの横画素数（領域の対称性より奇数に限定）
#define MODEL_HEIGHT    91      // モデルの縦画素数（領域の対称性より奇数に限定）
#define RADIUS_WIDTH    34      // 領域の横半径 (モデルの横画素数 - 1) / 2
#define RADIUS_HEIGHT   45      // 領域の縦半径 (モデルの縦画素数 - 1) / 2
#define Q_STEP          16      // 量子化間隔
#define Q_SIZE          16      // 量子化サイズ（RGB：256 / Q_STEP）
#define NBIN            4096    // ヒストグラムのビン数（RGB：Q_SIZE * Q_SIZE * QSIZE）
#define MAX_ITERATION   20      // Mean-Shift による最大更新回数

#define FRAME_NUM       100      //同時に処理する画像数

enum color {
    BLACK = 0,
    RED = 1,
    GREEN = 2,
    BLUE = 3,
    WHITE = 4
};

int COLOR_PALETTE[][3] = {
    {  0,   0,   0},
    {255,   0,   0},
    {  0, 255,   0},
    {  0,   0, 255},
    {255, 255, 255}
};


void calc_histogram(double hist[], unsigned char image[][MODEL_WIDTH][3],
                    double kernel[][MODEL_WIDTH]);
double calc_bhattacharrya(double hist1[], double hist2[]);
void calc_weight(double weight[][MODEL_WIDTH], unsigned char image[][MODEL_WIDTH][3],
                 double model_hist[], double candidate_hist[]);
void calc_mean_shift_vector(int *mx, int *my,
                            double weight[][MODEL_WIDTH], double kernel[][MODEL_WIDTH]);


void set_kernel(double kernel[][MODEL_WIDTH]);
void copy_region(unsigned char src[][WIDTH][3], unsigned char dst[][MODEL_WIDTH][3],
                 int upper_left_x, int upper_left_y);
void display_center_mark(unsigned char image[][WIDTH][3], int center_x, int center_y,
                         enum color color_index);
void display_rectangle(unsigned char image[][WIDTH][3],
                       int upper_left_x, int upper_left_y,
                       int width, int height, enum color color_index);
void load_ppm(unsigned char image[][WIDTH][3], const char path[]);
void save_ppm(unsigned char image[][WIDTH][3], const char path[]);
void load_model(unsigned char model[][MODEL_WIDTH][3], const char path[]);


int main(int argc, char *argv[]){
    unsigned char model_image[MODEL_HEIGHT][MODEL_WIDTH][3];
    unsigned char region_image[MODEL_HEIGHT][MODEL_WIDTH][3];
    unsigned char src_image[HEIGHT][WIDTH][3];
    unsigned char dst_image[HEIGHT][WIDTH][3];
    double kernel[MODEL_HEIGHT][MODEL_WIDTH];
    double weight[MODEL_HEIGHT][MODEL_WIDTH];
    double model_hist[NBIN];
    double candidate_hist[NBIN];
    double bhatt;
    int x, y;
    int ms_vector_x, ms_vector_y;
    int iteration;
    char buf[1024];


    // モデルの生成
    load_model(model_image, argv[1]);
    set_kernel(kernel);
    calc_histogram(model_hist, model_image, kernel);


    //  矩形の初期位置の設定（追跡対象のサイズは不変で，モデルのサイズと同一）
    //  Web のサンプルを完成させて実行すると，類似度最大となる矩形の左上位置は (362, 187)
    //  その位置から少しずれた位置から出発
    x = 340;
    y = 160;
    
   //100枚分の処理
   for(int frame = 0; frame < FRAME_NUM; frame++){
        sprintf(buf, "./src_image/sample%06d.ppm", frame);
        load_ppm(src_image, buf);              //原画像
        load_ppm(dst_image, buf);              //結果表示用の画像

        x = 340;
        y = 160;


        // 初期位置を青色で表示
        display_center_mark(dst_image, x + RADIUS_WIDTH, y + RADIUS_HEIGHT, BLUE);
        display_rectangle(dst_image, x, y, MODEL_WIDTH, MODEL_HEIGHT, BLUE);

        // 類似度が極大となる位置の探索を開始
        iteration = 0;
        do {
            // 矩形領域をコピーして，矩形領域のヒストグラムを計算
            copy_region(src_image, region_image, x, y);
            calc_histogram(candidate_hist, region_image, kernel);

            // 重みを計算
            calc_weight(weight, region_image, model_hist, candidate_hist);
            // 重みの重心を算出して移動量を計算
            calc_mean_shift_vector(&ms_vector_x, &ms_vector_y, weight, kernel);
            
            // 確認
            //printf("%2d %3d %3d\n", iteration, ms_vector_x, ms_vector_y);

            // 枠を移動する
            x += ms_vector_x;
            y += ms_vector_y;

            iteration++;

            // 探索途中の領域の中心を緑色で表示
            display_center_mark(dst_image, x + RADIUS_WIDTH, y + RADIUS_HEIGHT, GREEN);

        } while ( iteration < MAX_ITERATION && (ms_vector_x != 0 || ms_vector_y != 0) );

        display_center_mark(dst_image, x + RADIUS_WIDTH, y + RADIUS_HEIGHT, RED);
        display_rectangle(dst_image, x, y, MODEL_WIDTH, MODEL_HEIGHT, RED);
        sprintf(buf, "./result_image/result%06d.ppm", frame);
        save_ppm(dst_image, buf);
        printf("%sを保存しました\n", buf);
    }

    // 最終結果を赤色で表示
    display_center_mark(dst_image, x + RADIUS_WIDTH, y + RADIUS_HEIGHT, RED);
    display_rectangle(dst_image, x, y, MODEL_WIDTH, MODEL_HEIGHT, RED);
    save_ppm(dst_image, "result.ppm");

    return 0;
}

// ヒストグラムの計算
void calc_histogram(double hist[], unsigned char image[][MODEL_WIDTH][3],
                    double kernel[][MODEL_WIDTH]){
    int x, y;

    // ヒストグラムをゼロで初期化
    for(y = 0; y < NBIN; y++){
        hist[y] = 0;
    }

    // 重み付きヒストグラムを算出
    for (y = 0; y < MODEL_HEIGHT; y++) {
        for (x = 0; x < MODEL_WIDTH; x++) {

            // RGB値から色番号に変換
            int colorNum = (image[y][x][0] / Q_SIZE) + (image[y][x][1] / Q_SIZE) * Q_STEP
                            + (image[y][x][2] / Q_SIZE) * pow(Q_STEP, 2);

            // 色番号のビンに，カーネルの値をインクリメント
            hist[colorNum] += kernel[y][x];
        }
    }

    // ヒストグラムの正規化
    double histSum = 0;
    for(y = 0; y < NBIN; y++){
        histSum += hist[y];
    }
    for(y = 0; y < NBIN; y++){
        hist[y] /= histSum;
    }
}

// 重み計算
void calc_weight(double weight[][MODEL_WIDTH], unsigned char image[][MODEL_WIDTH][3],
                 double model_hist[], double candidate_hist[]){
    int x, y;

    for (y = 0; y < MODEL_HEIGHT; y++) {
        for (x = 0; x < MODEL_WIDTH; x++) {
            weight[y][x] = 0;
        }
    }

    // 追跡対象の候補領域の左上からラスタ操作順に，重みを計算
    for (y = 0; y < MODEL_HEIGHT; y++) {
        for (x = 0; x < MODEL_WIDTH; x++) {

            // RGB値から色番号に変換
            int colorNum = (image[y][x][0] / Q_SIZE) + (image[y][x][1] / Q_SIZE) * Q_STEP
                            + (image[y][x][2] / Q_SIZE) * pow(Q_STEP, 2);

            // 計算式に従って重みを算出
            //   スライドのように，色番号 u でループを回すのは非効率
            //   クロネッカーのデルタの性質を利用して効率化を図りましょう

            weight[y][x] += sqrt(model_hist[colorNum] / candidate_hist[colorNum]);
        }
    }
}


void calc_mean_shift_vector(int *mx, int *my,
                            double weight[][MODEL_WIDTH], double kernel[][MODEL_WIDTH]){
    int x, y;
    double num_x = 0.0;  // 分子の x 座標
    double num_y = 0.0;  // 分子の y 座標
    double dum = 0.0;    // 分母

    // 移動量の算出は，分子の x 座標と y 座標，分母に分けて累算すれば簡単
    for (y = 0; y < MODEL_HEIGHT; y++) {
        for (x = 0; x < MODEL_WIDTH; x++) {
            num_x += x * weight[y][x] * kernel[y][x];
            num_y += y * weight[y][x] * kernel[y][x];
            dum += weight[y][x] * kernel[y][x];
        }
    }

    // 上で算出した結果は，矩形の左上が原点なので，
    // 矩形の中心が原点になるように平行移動する
    *my = (int) round(num_y / dum) - RADIUS_HEIGHT;
    *mx = (int) round(num_x / dum) - RADIUS_WIDTH;
}

double calc_bhattacharrya(double hist1[], double hist2[]){
    // 正規化されたヒストグラムのバタッチャリヤ係数を計算
    double bhatt = 0.0;
    int y;

    for(y = 0; y < NBIN; y++){
        bhatt += sqrt(hist1[y] * hist2[y]);
    }

    return bhatt;
}

void set_kernel(double kernel[][MODEL_WIDTH]){
    int x, y;
    double xx, yy;
    double tmp;

    for (y = 0; y < MODEL_HEIGHT; y++) {
        for (x = 0; x < MODEL_WIDTH; x++) {
            xx = (x - RADIUS_WIDTH)  / (double) RADIUS_WIDTH;
            yy = (y - RADIUS_HEIGHT) / (double) RADIUS_HEIGHT;
            tmp = xx * xx + yy * yy;
            kernel[y][x] = exp(-tmp);
        }
    }
}


void copy_region(unsigned char src[][WIDTH][3], unsigned char dst[][MODEL_WIDTH][3],
                 int upper_left_x, int upper_left_y){
    int x, y;

    for (y = 0; y < MODEL_HEIGHT; y++) {
        for (x = 0; x < MODEL_WIDTH; x++) {
            dst[y][x][0] = src[upper_left_y + y][upper_left_x + x][0];
            dst[y][x][1] = src[upper_left_y + y][upper_left_x + x][1];
            dst[y][x][2] = src[upper_left_y + y][upper_left_x + x][2];
        }
    }
}

void display_center_mark(unsigned char image[][WIDTH][3], int center_x, int center_y,
                         enum color color_index){
    int x, y;

    for (y = -1; y <= 1; y++) {
        for (x = -1; x <= 1; x++) {
            image[center_y + y][center_x + x][0] = COLOR_PALETTE[color_index][0];
            image[center_y + y][center_x + x][1] = COLOR_PALETTE[color_index][1];
            image[center_y + y][center_x + x][2] = COLOR_PALETTE[color_index][2];
        }
    }
}

void display_rectangle(unsigned char image[][WIDTH][3],
                       int upper_left_x, int upper_left_y,
                       int width, int height,
                       enum color color_index){
    int x, y;

    for (y = upper_left_y; y < upper_left_y + height; y++) {
        image[y][upper_left_x][0] = COLOR_PALETTE[color_index][0];
        image[y][upper_left_x][1] = COLOR_PALETTE[color_index][1];
        image[y][upper_left_x][2] = COLOR_PALETTE[color_index][2];

        image[y][upper_left_x + width - 1][0] = COLOR_PALETTE[color_index][0];
        image[y][upper_left_x + width - 1][1] = COLOR_PALETTE[color_index][1];
        image[y][upper_left_x + width - 1][2] = COLOR_PALETTE[color_index][2];
    }

    for (x = upper_left_x; x < upper_left_x + width; x++) {
        image[upper_left_y][x][0] = COLOR_PALETTE[color_index][0];
        image[upper_left_y][x][1] = COLOR_PALETTE[color_index][1];
        image[upper_left_y][x][2] = COLOR_PALETTE[color_index][2];

        image[upper_left_y + height - 1][x][0] = COLOR_PALETTE[color_index][0];
        image[upper_left_y + height - 1][x][1] = COLOR_PALETTE[color_index][1];
        image[upper_left_y + height - 1][x][2] = COLOR_PALETTE[color_index][2];
    }
}

void load_model(unsigned char model[][MODEL_WIDTH][3], const char path[]){
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
    if (magic_number[0] != 'P' || magic_number[1] != '6') {
        fprintf(stderr, "%s はバイナリ型 PPM ではありません．\n", path);
        fclose(fp);
        exit(1);
    }

    fscanf(fp, "%d %d", &width, &height);
    if (width != MODEL_WIDTH || height != MODEL_HEIGHT) {
        fprintf(stderr, "画像のサイズが異なります．\n");
        fprintf(stderr, "  想定サイズ：WIDTH = %d, HEIGHT = %d\n", MODEL_WIDTH, MODEL_HEIGHT);
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

    fread(model, sizeof(unsigned char), MODEL_HEIGHT * MODEL_WIDTH * 3, fp);

    fclose(fp);
}


void load_ppm(unsigned char image[][WIDTH][3], const char path[]){
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
    if (magic_number[0] != 'P' || magic_number[1] != '6') {
        fprintf(stderr, "%s はバイナリ型 PPM ではありません．\n", path);
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

    fread(image, sizeof(unsigned char), HEIGHT * WIDTH * 3, fp);

    fclose(fp);
}

void save_ppm(unsigned char image[][WIDTH][3], const char path[]){
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
