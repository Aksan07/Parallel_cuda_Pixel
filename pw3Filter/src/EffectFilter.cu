/*
 * EffectFilter.cu
 *
 * CUDA port of effectFilter.cl (Parallel Worlds 3).
 */

#include "EffectFilter.hpp"

// helper macros (same idea as in the .cl file)
// ARR  : index into a 1-channel (grey) image
// ARRC : index into a 3-channel (colour, BGR) image, picking one channel
#define ARR(A,x,y,maxX)          ((A)[(x)+(y)*(maxX)])
#define ARRC(A,x,y,maxX,channel) ((A)[((x)+(y)*(maxX))*3+(channel)])

/// Actual kernel of the effect filter.
// @param[in]  colImg  colour image (input, 3 bytes/pixel, BGR)
// @param[in]  edgeImg grey edge image from Sobel (input, 1 byte/pixel)
// @param[out] outImg  colour image (output, 3 bytes/pixel)
// @param[in]  w       image width
// @param[in]  h       image height
// @param[in]  threshold edge strength above which a pixel is darkened
__global__ void effectKernel(const unsigned char *colImg,
		const unsigned char *edgeImg, unsigned char *outImg,
		const unsigned int w, const unsigned int h, const float threshold) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		// read the edge strength at this pixel
		float G    = (float) ARR(edgeImg, x, y, w);
		float absG = fabsf(G);

		// copy the three colour channels through, darkening them
		// (multiply by 0.5) wherever the edge is strong enough
		float factor = (absG > threshold) ? 0.5f : 1.0f;
		ARRC(outImg, x, y, w, 0) = (unsigned char)(ARRC(colImg, x, y, w, 0) * factor);
		ARRC(outImg, x, y, w, 1) = (unsigned char)(ARRC(colImg, x, y, w, 1) * factor);
		ARRC(outImg, x, y, w, 2) = (unsigned char)(ARRC(colImg, x, y, w, 2) * factor);
	}
}

/// Host wrapper: does the copy-in / launch / copy-out dance.
// Unlike grey/sobel there are TWO inputs to upload (colour + edge).
__host__ void EffectFilter::operator()(const unsigned char *input,
		const unsigned char *edgeInput, unsigned char *output,
		const unsigned int w, const unsigned int h, const float threshold) {
	this->prepareBuffers(w, h);

	// upload the colour image -> dInput
	SAFE_CALL(cudaMemcpy(this->dInput, reinterpret_cast<const void*>(input),
			w * h * this->depthIn, cudaMemcpyHostToDevice));
	// upload the grey edge image -> dEdge
	SAFE_CALL(cudaMemcpy(this->dEdge, reinterpret_cast<const void*>(edgeInput),
			w * h * this->depthEdge, cudaMemcpyHostToDevice));

	// one thread per pixel (simple grid, like the grey filter)
	effectKernel<<<this->grid, this->threads>>>(dInput, dEdge, dOutput, w, h, threshold);

	// download the colour result <- dOutput
	SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(output), this->dOutput,
			w * h * this->depthOut, cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaDeviceSynchronize());
}
