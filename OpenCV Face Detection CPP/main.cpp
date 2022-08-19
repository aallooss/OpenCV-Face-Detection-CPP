#include <opencv2\opencv.hpp>
#include <vector>


using namespace cv;
using namespace std;

int main()
{
	double scale = 3.0;

	CascadeClassifier faceCascade;
	faceCascade.load("C:\\OpenCV4\\etc\\haarcascades\\haarcascade_frontalface_alt.xml");
	CascadeClassifier eyeCascade;
	eyeCascade.load("C:\\OpenCV4\\etc\\haarcascades\\haarcascade_eye.xml");
	CascadeClassifier smileCascade;
	smileCascade.load("C:\\OpenCV4\\etc\\haarcascades\\haarcascade_smile.xml");


	VideoCapture cap(0);
	if (!cap.isOpened())
		return -1;
	
	for (;;)
	{
		Mat frame;
		cap >> frame;

		Mat grayscale;
		cvtColor(frame, grayscale, COLOR_BGR2GRAY);
		resize(grayscale, grayscale, Size(grayscale.size().width / scale, grayscale.size().height / scale));

		vector<Rect> faces;
		faceCascade.detectMultiScale(grayscale, faces, 1.1, 3, 0, Size(30, 30));
		vector<Rect> eye;
		eyeCascade.detectMultiScale(grayscale, eye, 1.1, 3, 0);
		vector<Rect> smile;
		smileCascade.detectMultiScale(grayscale, smile, 1.1, 60, 0, Size(30,30), Size(150,150));

		for (Rect area : faces)
		{
			Scalar drawColor = Scalar(255, 0, 0);
			rectangle(frame, Point(cvRound(area.x * scale), cvRound(area.y * scale)),
				Point(cvRound((area.x + area.width - 1) * scale), cvRound((area.y + area.height - 1) * scale)), drawColor);
			for (Rect area : eye)
			{
				Scalar drawColor = Scalar(0, 255, 0);
				rectangle(frame, Point(cvRound(area.x * scale), cvRound(area.y * scale)),
					Point(cvRound((area.x + area.width - 1) * scale), cvRound((area.y + area.height - 1) * scale)), drawColor);
			}
			for (Rect area : smile)
			{
				Scalar drawColor = Scalar(0, 0, 255);
				rectangle(frame, Point(cvRound(area.x * scale), cvRound(area.y * scale)),
					Point(cvRound((area.x + area.width - 1) * scale), cvRound((area.y + area.height - 1) * scale)), drawColor);
			}
		}

		imshow("Webcam Frame", frame);

		if (waitKey(30) >= 0)
			break;
	}
	return 0;
}