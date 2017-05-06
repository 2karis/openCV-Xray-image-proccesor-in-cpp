#include "opencv2\core.hpp";
#include "opencv2\imgproc.hpp";
#include "opencv2\highgui.hpp";
#include "opencv2\photo.hpp";
#include "iostream";

char window_name[] = " Demo";
int lowThreshold = 86;
cv::Mat fingers,dst;

void detect_hand_and_fingers(cv::Mat& src);
void detect_hand_silhoutte(cv::Mat& src);
void CannyThreshold(int, void*);
void detect_edge();

int main(int argc, char* argv[])
{
	cv::Mat img = cv::imread(argv[1]);
	if (img.empty())
	{
		std::cout << "!!! imread() failed to open target image" << std::endl;
		return -1;
	}

	// Convert RGB Mat to GRAY
	cv::Mat gray, dnsed;
	cv::cvtColor(img, gray, CV_BGR2GRAY);

	cv::fastNlMeansDenoising(gray, gray, 3, 7, 21);

	cv::Mat gray_silhouette = gray.clone();

	//gray = dnsed.clone();

	/* Isolate Hand + Fingers */

	detect_hand_and_fingers(gray);
	cv::imshow("Hand+Fingers", gray);
	cv::imwrite("hand_fingers.png", gray);

	/* Isolate Hand Sillhoute and subtract it from the other image (Hand+Fingers) */

	detect_hand_silhoutte(gray_silhouette);
	cv::imshow("Hand", gray_silhouette);
	cv::imwrite("hand_silhoutte.png", gray_silhouette);

	/* Subtract Hand Silhoutte from Hand+Fingers so we get only Fingers */

	fingers = gray - gray_silhouette;
	cv::imshow("Fingers", fingers);
	cv::imwrite("fingers_only.png", fingers);
	cv::waitKey(0);
	return 0;
}

void detect_hand_and_fingers(cv::Mat& src)
{
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1));
	cv::morphologyEx(src, src, cv::MORPH_ELLIPSE, kernel);

	int adaptiveMethod = CV_ADAPTIVE_THRESH_GAUSSIAN_C; // CV_ADAPTIVE_THRESH_MEAN_C, CV_ADAPTIVE_THRESH_GAUSSIAN_C
	cv::adaptiveThreshold(src, src, 255,
		adaptiveMethod, CV_THRESH_BINARY,
		9, -9);

	int dilate_sz = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
		cv::Size(2 * dilate_sz, 2 * dilate_sz),
		cv::Point(dilate_sz, dilate_sz));
	cv::dilate(src, src, element);
}

void detect_hand_silhoutte(cv::Mat& src)
{
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(3, 3));
	cv::morphologyEx(src, src, cv::MORPH_ELLIPSE, kernel);

	int adaptiveMethod = CV_ADAPTIVE_THRESH_MEAN_C; // CV_ADAPTIVE_THRESH_MEAN_C, CV_ADAPTIVE_THRESH_GAUSSIAN_C
	cv::adaptiveThreshold(src, src, 255,
		adaptiveMethod, CV_THRESH_BINARY,
		251, 9); // 251, 5

	int erode_sz = 5;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
		cv::Size(2 * erode_sz + 1, 2 * erode_sz + 1),
		cv::Point(erode_sz, erode_sz));
	cv::erode(src, src, element);

	int dilate_sz = 1;
	element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
		cv::Size(2 * dilate_sz + 1, 2 * dilate_sz + 1),
		cv::Point(dilate_sz, dilate_sz));
	cv::dilate(src, src, element);

	cv::bitwise_not(src, src);
}

void CannyThreshold(int, void*)
{
	cv::Canny(fingers, dst, lowThreshold, lowThreshold*3, 3);
	cv::imshow(window_name, dst);
}

void detect_edge()
{
	cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	cv::createTrackbar("Threshold:", window_name, &lowThreshold, 100, CannyThreshold);
	CannyThreshold(0, 0);
}