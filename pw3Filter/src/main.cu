#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdio>

#include "CUDAInterface.hpp"
#include "GreyFilter.hpp"
#include "SobelFilter.hpp"
#include "EffectFilter.hpp"
#include "PixelFilter.hpp"

//                       depthIn, depthOut
GreyFilter   greyFilter(3, 1);
SobelFilter  sobelFilter(1, 1);
//                        depthIn, depthEdge, depthOut
EffectFilter effectFilter(3, 1, 3);
//                      depthIn, depthOut, initial tile size N
PixelFilter  pixelFilter(3, 3, 8);

/// Run the pixelization filter at several tile sizes and print a timing
/// comparison using the kernel times measured via CUDA Events.
__host__ void runPixelBenchmark(const cv::Mat &frame, unsigned int w, unsigned int h) {
	const unsigned int sizes[] = {8, 16, 32, 64};
	const int numSizes = 4;
	const int repeats = 20; // average several runs for a stable number

	cv::Mat outFrame(h, w, CV_8UC3);

	std::printf("\n==================== Pixelization timing ====================\n");
	std::printf(" image: %u x %u   (%u pixels)   averaged over %d runs\n",
				w, h, w * h, repeats);
	std::printf("------------------------------------------------------------\n");
	std::printf(" %-10s | %-14s | %-22s\n", "tile N", "kernel (ms)", "note");
	std::printf("------------------------------------------------------------\n");

	float baseline = 0.0f;
	for (int i = 0; i < numSizes; ++i) {
		pixelFilter.setTileSize(sizes[i]);

		// warm-up run (first launch includes one-time setup cost)
		pixelFilter(frame.data, outFrame.data, w, h);

		// timed runs
		float total = 0.0f;
		for (int r = 0; r < repeats; ++r)
			total += pixelFilter(frame.data, outFrame.data, w, h);
		float avg = total / repeats;

		if (i == 0) baseline = avg;

		char note[64];
		if (i == 0) std::snprintf(note, sizeof(note), "fastest (baseline)");
		else        std::snprintf(note, sizeof(note), "%.2fx slower than N=8", avg / baseline);

		std::printf(" %-10u | %-14.4f | %-22s\n", sizes[i], avg, note);
	}
	std::printf("============================================================\n");
	std::printf(" Note: bigger tiles average more pixels per thread, so the\n");
	std::printf(" kernel does more work and takes longer. Unlike the naive\n");
	std::printf(" one-thread-per-tile design, N=64 runs here because the\n");
	std::printf(" CUDA block is a fixed 16x16 (256 threads, under the 1024 limit).\n");
	std::printf("============================================================\n\n");
}

__host__ int main(int argc, const char** argv) {
	cv::VideoCapture capture(0); //0=default, -1=any camera, 1..99=your camera
	cv::Mat frame;

	bool cameraOn = capture.isOpened() && false; // force preview.png as in PW2/PW3
	if (cameraOn) {
		if (!capture.read(frame))
			exit(3);
	} else {
		std::cerr << "No camera detected" << std::endl;
		frame = cv::imread("preview.png");
		if (frame.data == NULL)
			exit(3);
	}

	const unsigned int w = frame.cols;
	const unsigned int h = frame.rows;

	cv::Mat convertedFrame(h, w, CV_8UC1); // grey
	cv::Mat edgeFrame(h, w, CV_8UC1);      // edges
	cv::Mat effectFrame(h, w, CV_8UC3);    // shaded-edge effect (colour)
	cv::Mat pixelFrame(h, w, CV_8UC3);     // pixelization / mosaic (colour)

	// --- run the timing comparison once, up front, and print results ---
	runPixelBenchmark(frame, w, h);

	// for the live display, use a fixed mosaic size
	pixelFilter.setTileSize(16);

	cv::namedWindow("preview", 0);
	cv::namedWindow("converted", 0);
	cv::namedWindow("edge", 0);
	cv::namedWindow("effect", 0);
	cv::namedWindow("pixelized", 0);

	while (((char)cv::waitKey(10)) <= -1) {
		if (cameraOn && !capture.read(frame))
			exit(3);

		greyFilter(frame.data, convertedFrame.data, w, h);
		sobelFilter(convertedFrame.data, edgeFrame.data, w, h, .5f);
		effectFilter(frame.data, edgeFrame.data, effectFrame.data, w, h, 90.f);
		pixelFilter(frame.data, pixelFrame.data, w, h);

		cv::imshow("preview", frame);
		cv::imshow("converted", convertedFrame);
		cv::imshow("edge", edgeFrame);
		cv::imshow("effect", effectFrame);
		cv::imshow("pixelized", pixelFrame);
	}

	cv::destroyWindow("preview");
	cv::destroyWindow("converted");
	cv::destroyWindow("edge");
	cv::destroyWindow("effect");
	cv::destroyWindow("pixelized");

	return 0;
}
