/*　3807024K 鍋島虎太郎
 * 
 *  Horn＆Schunckの方法でフロー計算
 *  2枚のPGM画像を読み込み，そのフローを"flow.txt"というファイルに保存
 *
 * 　(コンパイル・実行例)
 *   gcc 3807024k_flow.c
 *   .\a.exe frame_0000.pgm frame_0001.pgm
 *
 *   (実行環境)
 *   gcc version 12.2.0
 *   windows 11
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define HEIGHT 576
#define WIDTH 768
#define IPR_GRAD_H 0            //水平方向
#define IPR_GRAD_V 1            //垂直方向
#define LAMDA 1                 //lambdaを指定
#define ITERATION_NUM 20        //反復回数を指定

void load_pgm(unsigned char image[][WIDTH], const char path[]);
//void save_pgm(unsigned char image[][WIDTH], const char path[]);

void ipr_sobel(double **sobel, unsigned char image[][WIDTH], int direction);
void save_flow(double **flow_V, double **flow_H, const char path[]);

void time_direction_differentiation(double **diff_time,
                                    unsigned char image_pre[][WIDTH], unsigned char image_now[][WIDTH]);
void flow_Horn_Schunck(double **sobel_X, double **sobel_Y, double **time_diff,
                        double **u, double **v);
void iteration_horn_schunck(double **sobel_X, double **sobel_Y, double **time_diff,
                            double **u_old, double **v_old, double **u_new, double **v_new);


double average_local(double **num, int i, int j);
void copy_u_v(double **u_to, double **v_to, double **u_from, double **v_from);

int main(int argc, char *argv[]){

    unsigned char src_img[HEIGHT][WIDTH];
    unsigned char dst_img[HEIGHT][WIDTH];

    double **sobel_X, **sobel_Y, **time_diff, **flow_u, **flow_v;   //横がu,縦がv
    sobel_X = (double**)malloc(sizeof(double*)*HEIGHT);          //横方向微分
    sobel_Y = (double**)malloc(sizeof(double*)*HEIGHT);          //縦方向微分
    time_diff = (double**)malloc(sizeof(double*)*HEIGHT);        //時間方向微分
    flow_u = (double**)malloc(sizeof(double*)*HEIGHT);           //フロー u
    flow_v = (double**)malloc(sizeof(double*)*HEIGHT);           //フロー v

    for(int i = 0; i < HEIGHT; i++){                         //2次元配列確保
        sobel_X[i] = (double*)malloc(sizeof(double)*WIDTH);
        sobel_Y[i] = (double*)malloc(sizeof(double)*WIDTH);
        time_diff[i] = (double*)malloc(sizeof(double)*WIDTH);
        flow_u[i] = (double*)malloc(sizeof(double)*WIDTH);
        flow_v[i] = (double*)malloc(sizeof(double)*WIDTH);
    }

    if (argc < 3) {
        fprintf(stderr, "Usage: %s source destination\n", argv[0]);
        exit(1);
    }

    // ２枚の画像を読み込み
    load_pgm(src_img, argv[1]);
    load_pgm(dst_img, argv[2]);

    ipr_sobel(sobel_Y, src_img, IPR_GRAD_H);      //水平方向のソーベルフィルタ
    ipr_sobel(sobel_X, src_img, IPR_GRAD_V);      //垂直方向のソーベルフィルタ

    time_direction_differentiation(time_diff, src_img, dst_img); //時間方向の微分

    flow_Horn_Schunck(sobel_X, sobel_Y, time_diff, flow_u, flow_v); //フロー計算

    save_flow(flow_v, flow_u, "flow.txt");         //flow.txt に保存

    for(int i = 0; i < HEIGHT; i++){               //メモリ解放
        free(sobel_X[i]);
        free(sobel_Y[i]);
        free(time_diff[i]);
        free(flow_u[i]);
        free(flow_v[i]);
    }

    return 0;
}

// PGM画像を保存
void load_pgm(unsigned char image[][WIDTH], const char path[]){
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

//時間方向微分
void time_direction_differentiation(double **diff_time,
                                    unsigned char image_pre[][WIDTH], unsigned char image_now[][WIDTH]){

    for(int i = 0; i < HEIGHT; i++){
        for(int j = 0; j < WIDTH; j++){
            diff_time[i][j] = image_now[i][j] - image_pre[i][j];
        }
    }
}

//Horn＆Schunckの方法でフローを計算
void flow_Horn_Schunck(double **sobel_X, double **sobel_Y, double **time_diff,
                        double **u, double **v){

    int i, j;
    double **u_update, **v_update;

    u_update = (double**)malloc(sizeof(double*)*HEIGHT);     //uを更新した値を一時的に格納
    v_update = (double**)malloc(sizeof(double*)*HEIGHT);     //vを更新した値を一時的に格納

    for(int i = 0; i < HEIGHT; i++){                         //2次元配列確保
        u_update[i] = (double*)malloc(sizeof(double)*WIDTH);
        v_update[i] = (double*)malloc(sizeof(double)*WIDTH);
    }

    for(i = 0; i < HEIGHT; i++){                              //u,vの初期値を設定
        for(j = 0; j < WIDTH; j++){
            u[i][j] = 0;
            v[i][j] = 0;
        }
    }

    for(i = 0; i < ITERATION_NUM; i++){                    //何度か反復してu,vを得る
        //新たなuvを求める
        iteration_horn_schunck(sobel_X, sobel_Y, time_diff, u, v, u_update, v_update);
        //求めたuvで更新する
        copy_u_v(u, v, u_update, v_update);
    }

    for(int i = 0; i < HEIGHT; i++){                       //メモリ解放
        free(u_update[i]);
        free(v_update[i]);
    }

}

//反復式を計算
void iteration_horn_schunck(double **sobel_X, double **sobel_Y, double **time_diff,
                            double **u_old, double **v_old, double **u_new, double **v_new){

    for(int i = 1; i < HEIGHT - 1; i++){         //画像の端は常に0
        for(int j = 1; j < WIDTH - 1; j++){
            double u_ave, v_ave;                 //u,vの局所平均
            u_ave = average_local(u_old, i, j);
            v_ave = average_local(v_old, i, j);

            //ヤコビ法から求められた更新式
            u_new[i][j] = u_ave - ((sobel_X[i][j] * u_ave + sobel_Y[i][j] + time_diff[i][j])
                                    / (1 + LAMDA * (pow(sobel_X[i][j], 2.0) + pow(sobel_Y[i][j], 2.0)))) * sobel_X[i][j];
            v_new[i][j] = v_ave - ((sobel_X[i][j] * u_ave + sobel_Y[i][j] + time_diff[i][j])
                                    / (1 + LAMDA * (pow(sobel_X[i][j], 2.0) + pow(sobel_Y[i][j], 2.0)))) * sobel_Y[i][j];
        }
    }
}

//局所平均の計算
double average_local(double **num, int i, int j){
    return (num[i+1][j] + num[i][j-1] + num[i-1][j] + num[i][j+1]) / 4;
}

//新しいu,vを格納する
void copy_u_v(double **u_to, double **v_to, double **u_from, double **v_from){
    for(int i = 1; i < HEIGHT - 1; i++){
        for(int j = 1; j < WIDTH - 1; j++){
            u_to[i][j] = u_from[i][j];
            v_to[i][j] = v_from[i][j];
        }
    }
}

//フロー結果をflow.txt に保存
void save_flow(double **flow_V, double **flow_H, const char path[]){
    FILE *fp;

    fp = fopen(path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "%s が開けませんでした．\n", path);
        exit(1);
    }

    fprintf(fp, "# 横座標 縦座標 水平方向フロー 垂直方向フロー\n");

    for(int i = 0; i < HEIGHT; i++){
        for(int j = 0; j < WIDTH; j++){
            fprintf(fp, "%3d %3d  %.3lf %.3lf\n", j, i, flow_H[i][j], flow_V[i][j]);
        }
    }

    fclose(fp);
}
