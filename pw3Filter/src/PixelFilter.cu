/*
 * PixelFilter.cu
 *
 * CUDA pixelization (mosaic) filter with CUDA-Event kernel timing.
 */

#include "PixelFilter.hpp"

// index into a 3-channel (BGR) image, picking one channel
#define ARRC(A,x,y,maxX,channel) ((A)[((x)+(y)*(maxX))*3+(channel)])

/// Mosaic kernel: each thread owns ONE output pixel. It finds the top-left
// corner of its N x N tile, averages every pixel in that tile (clamped to
// the image bounds), and writes the average to its own pixel.
// @param inImg  colour input  (3 bytes/pixel, BGR)
// @param outImg colour output (3 bytes/pixel, BGR)
// @param w      image width
// @param h      image height
// @param n      mosaic tile edge length
__global__ void pixelKernel(const unsigned char *inImg, unsigned char *outImg,
		const unsigned int w, const unsigned int h, const unsigned int n) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		// top-left corner of the mosaic tile this pixel belongs to
		unsigned int tileX = (x / n) * n;
		unsigned int tileY = (y / n) * n;

		// accumulate the three colour channels over the tile
		unsigned int sumB = 0, sumG = 0, sumR = 0, count = 0;
		for (unsigned int dy = 0; dy < n; ++dy) {
			unsigned int sy = tileY + dy;
			if (sy >= h) break;            // tile runs off the bottom edge
			for (unsigned int dx = 0; dx < n; ++dx) {
				unsigned int sx = tileX + dx;
				if (sx >= w) break;        // tile runs off the right edge
				sumB += ARRC(inImg, sx, sy, w, 0);
				sumG += ARRC(inImg, sx, sy, w, 1);
				sumR += ARRC(inImg, sx, sy, w, 2);
				++count;
			}
		}

		// write the tile average to this pixel
		ARRC(outImg, x, y, w, 0) = (unsigned char)(sumB / count);
		ARRC(outImg, x, y, w, 1) = (unsigned char)(sumG / count);
		ARRC(outImg, x, y, w, 2) = (unsigned char)(sumR / count);
	}
}

/// Host wrapper. Does the copy-in / launch / copy-out dance and times ONLY
// the kernel itself using CUDA Events (high-precision GPU timestamps).
// Returns the elapsed kernel time in milliseconds.
__host__ float PixelFilter::operator()(const unsigned char *input,
		unsigned char *output, const unsigned int w, const unsigned int h) {
	this->prepareBuffers(w, h);

	SAFE_CALL(cudaMemcpy(this->dInput, reinterpret_cast<const void*>(input),
			w * h * this->depthIn, cudaMemcpyHostToDevice));

	// --- CUDA Event timing around the kernel launch only ---
	cudaEvent_t start, stop;
	SAFE_CALL(cudaEventCreate(&start));
	SAFE_CALL(cudaEventCreate(&stop));

	SAFE_CALL(cudaEventRecord(start));
	pixelKernel<<<this->grid, this->threads>>>(dInput, dOutput, w, h, this->tileSize);
	SAFE_CALL(cudaEventRecord(stop));

	// wait for the kernel + measure
	SAFE_CALL(cudaEventSynchronize(stop));
	float ms = 0.0f;
	SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));

	SAFE_CALL(cudaEventDestroy(start));
	SAFE_CALL(cudaEventDestroy(stop));
	// -------------------------------------------------------

	SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(output), this->dOutput,
			w * h * this->depthOut, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaDeviceSynchronize());

	return ms;
}
