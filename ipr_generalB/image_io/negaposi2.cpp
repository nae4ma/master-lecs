/* 画像のネガポジ変換を行うプログラム*/
/*二重ループとbitwise_notを使用した変換を行い，時間を計測*/
/*実行環境：visual studio 2022(community)，opencv 4.7.0*/

/*コマンドライン引数より，画像名.png が入力されることを想定する*/

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

// OpenCV のバージョンが違うなら名前は 454 の部分を自身のものに変更する
#ifdef _DEBUG
#pragma comment(lib, "opencv_world470d.lib")
#else
#pragma comment(lib, "opencv_world470.lib")
#endif

int main(int argc, char* argv[])
{

    // 画像パス
    cv::Mat img = cv::imread(argv[1]);

    //変換後の画像
    cv::Mat img_np_loop;
    cv::Mat img_np_bwn;

    //chronoによる時間計測
    std::chrono::system_clock::time_point  start, end;
    std::time_t time_stamp;

    //二重ループによるネガポジ変換
    start = std::chrono::system_clock::now();   //時間計測開始
    if (img.channels() == 3) {
        //変換後の画像img_npを初期化
        img_np_loop = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);   //CV_8UC3: 1個のCV_8U モノクロ画像の初期値

        //二重ループでネガポジ変換
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img_np_loop.at<cv::Vec3b>(i, j)[0] = 255 - img.at<cv::Vec3b>(i, j)[0];   //B
                img_np_loop.at<cv::Vec3b>(i, j)[1] = 255 - img.at<cv::Vec3b>(i, j)[1];   //G
                img_np_loop.at<cv::Vec3b>(i, j)[2] = 255 - img.at<cv::Vec3b>(i, j)[2];   //R
            }
        }
    }
    else if (img.channels() == 1) {
        //変換後の画像img_npを初期化
        img_np_loop = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);  //CV_8UC1: 1個のCV_8U モノクロ画像の初期値

        //二重ループでネガポジ変換
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img_np_loop.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
            }
        }
    }
    end = std::chrono::system_clock::now(); //時間計測終了
    auto time_loop = end - start;

    // 処理に要した時間をミリ秒に変換
    auto msec_loop = std::chrono::duration_cast<std::chrono::milliseconds>(time_loop).count();
    std::cout << "use_loop: " << msec_loop << " msec" << std::endl;  //bitwise_notの処理時間

    //bitwise_notによるネガポジ変換
    start = std::chrono::system_clock::now();   //時間計測開始
    if (img.channels() == 3) {
        //変換後の画像img_npを初期化
        img_np_bwn = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);

        cv::bitwise_not(img, img_np_bwn);

    }else if (img.channels() == 1) {
        //変換後の画像img_npを初期化
        img_np_bwn = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

        cv::bitwise_not(img, img_np_bwn);
    }

    end = std::chrono::system_clock::now(); //時間計測終了
    auto time_bwn = end - start;

    // 処理に要した時間をミリ秒に変換
    auto msec_bwn = std::chrono::duration_cast<std::chrono::milliseconds>(time_bwn).count();
    std::cout << "use_bitwise_not: " << msec_bwn << " msec" << std::endl;   //bitwise_notの処理時間


    //画像の出力
    cv::imwrite("loop_np.png", img_np_loop);
    cv::imwrite("bitwise_not_np.png", img_np_bwn);

    cv::imshow("use_loop", img_np_loop);
    cv::imshow("use_bitwise_not", img_np_bwn);
    cv::waitKey(0);

    return 0;
}
