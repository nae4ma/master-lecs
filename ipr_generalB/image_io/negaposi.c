/* 画像にネガポジ変換を行い，出力するプログラム*/
/* load_PGM,save_PGMでモノクロ画像，load_PPM,save_PPMでカラー画像を処理する*/
/*モノクロ画像はgray_np.pgm，カラー画像はcolor_np.ppm で出力される*/
/* 実行環境: gcc 9.2.0*/

/*コマンドライン引数から，画像名.ppm(画像名.pgm)が入力されることを想定*/

#include <stdio.h>
#include <stdlib.h>     /* 異常終了 exit() で必要 */
#include <string.h>     /*拡張子判定*/

#define HEIGHT 256     /* 縦画素数 */
#define WIDTH  256     /* 横画素数 */
#define mode 2         /* PPMかPGMか指定*/

/* 関数のプロトタイプ宣言 */
void load_PGM(unsigned char image[][WIDTH], const char *filename);
void save_PGM(unsigned char image[][WIDTH], const char *filename);

void load_PPM(unsigned char image[][WIDTH][3], const char *filename);
void save_PPM(unsigned char image[][WIDTH][3], const char *filename);

int main(int argc, char *argv[])
{
    
    if(mode==1){    //mode=1:ppm
        unsigned char src_image[HEIGHT][WIDTH][3]; 
        load_PPM(src_image, argv[1]);
        save_PPM(src_image, argv[1]);
    }

    if(mode==2){     //mode=2:pgm
        unsigned char src_image[HEIGHT][WIDTH]; 
        load_PGM(src_image, argv[1]);
        save_PGM(src_image, argv[1]); 
    }

    return 0;
}

/* 関数load_PGM: 二次元配列image に格納*/
void load_PGM(unsigned char image[][WIDTH], const char *filename){

    char magic_number[2];
    int height, width;
    int max_intensity;
    FILE *fp;

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "画像 %s が開けませんでした．\n", filename);
        exit(1);
    }

    /* マジックナンバを読込んでチェック */
    fscanf(fp, "%c%c", &magic_number[0], &magic_number[1]);
    if (magic_number[0] != 'P' || magic_number[1] != '5') {
        fprintf(stderr, "画像 %s はバイナリ形式の PGM ではありません．", filename);
	exit(1);
    }

    /* 画像のサイズを読込んでチェック */
    fscanf(fp, "%d %d", &width, &height);
    if (width != WIDTH || height != HEIGHT) {
        fprintf(stderr, "サイズが異なります．\n");
	exit(1);
    }

    /* 最大階調値を読込んでチェック */
    fscanf(fp, "%d", &max_intensity);
    if (max_intensity != 255) {
        fprintf(stderr, "最大階調値が範囲外です．\n");
        exit(1);
    }
    fgetc(fp); /* ヘッダと画像データの区切りである改行文字の読み飛ばし */

    /* 画像データの読込み */
    fread(image, sizeof(unsigned char), HEIGHT * WIDTH, fp);

    fclose(fp);
}

/* save_PGM: 入力画像のネガポジ変換を行い，pgmファイルを保存する*/
void save_PGM(unsigned char image[][WIDTH], const char *filename){

    char magic_number[2];
    int height, width;
    int max_intensity;
    FILE *fp;

    /* ヘッダ情報の読み出し*/
    fp = fopen(filename, "rb");

    fscanf(fp, "%c%c", &magic_number[0], &magic_number[1]);
    fscanf(fp, "%d %d", &width, &height);
    fscanf(fp, "%d", &max_intensity);

    /* ネガポジ変換*/
    for(int h=0;h<HEIGHT;h++){
        for(int w=0;w<WIDTH;w++){
            image[h][w] = 255 - image[h][w];
        }
    }

    /* ヘッダ情報の書き出し*/
    fp = fopen("gray_np.pgm", "wb");

    fprintf(fp, "%c%c\n", magic_number[0], magic_number[1]); 
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "%d\n", max_intensity);
    fwrite(image, sizeof(unsigned char), HEIGHT * WIDTH, fp); /* 画像データの書き出し */
    fclose(fp); 

}

/* 関数load_PPM: 三次元配列image に格納*/
void load_PPM(unsigned char image[][WIDTH][3], const char *filename){

    char magic_number[2];
    int height, width;
    int max_intensity;
    FILE *fp;

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "画像 %s が開けませんでした．\n", filename);
        exit(1);
    }

    /* マジックナンバを読込んでチェック */
    fscanf(fp, "%c%c", &magic_number[0], &magic_number[1]);
    if (magic_number[0] != 'P' || magic_number[1] != '6') {
        fprintf(stderr, "画像 %s はバイナリ形式の PPM ではありません．", filename);
	exit(1);
    }

    /* 画像のサイズを読込んでチェック */
    fscanf(fp, "%d %d", &width, &height);
    if (width != WIDTH || height != HEIGHT) {
        fprintf(stderr, "サイズが異なります．\n");
	exit(1);
    }

    /* 最大階調値を読込んでチェック */
    fscanf(fp, "%d", &max_intensity);
    if (max_intensity != 255) {
        fprintf(stderr, "最大階調値が範囲外です．\n");
        exit(1);
    }
    fgetc(fp); /* ヘッダと画像データの区切りである改行文字の読み飛ばし */

    /* 画像データの読込み */
    fread(image, sizeof(unsigned char), HEIGHT * WIDTH * 3, fp);

    fclose(fp);
}

/* save_PPM: 入力画像のネガポジ変換を行い，ppmファイルを保存する*/
void save_PPM(unsigned char image[][WIDTH][3], const char *filename){

    char magic_number[2];
    int height, width;
    int max_intensity;
    FILE *fp;

    /* ヘッダ情報の読み出し*/
    fp = fopen(filename, "rb");

    fscanf(fp, "%c%c", &magic_number[0], &magic_number[1]);
    fscanf(fp, "%d %d", &width, &height);
    fscanf(fp, "%d", &max_intensity);

    /* ネガポジ変換*/
    for(int h=0;h<HEIGHT;h++){
        for(int w=0;w<WIDTH;w++){
            for (int rgb_num=0;rgb_num<3;rgb_num++){
                image[h][w][rgb_num] = 255 - image[h][w][rgb_num];
            }
        }
    }

    /* ヘッダ情報の書き出し*/
    fp = fopen("color_np.ppm", "wb");

    fprintf(fp, "%c%c\n", magic_number[0], magic_number[1]); 
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "%d\n", max_intensity);
    fwrite(image, sizeof(unsigned char), HEIGHT * WIDTH*3, fp); /* 画像データの書き出し */
    fclose(fp); 

}
