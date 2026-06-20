/*
 * PixelFilter.hpp
 *
 * CUDA pixelization (mosaic) filter — the project's own extension.
 *
 * Divides the image into N x N mosaic tiles and replaces every pixel
 * in a tile with the AVERAGE colour of that tile (a mosaic / censoring
 * / retro "pixel" look).
 *
 * Design note: this is "one thread per pixel". Each thread works out
 * which mosaic tile its pixel belongs to, averages that tile, and writes
 * its own output pixel. The CUDA thread-block stays a fixed 16x16 no
 * matter how big the mosaic tile N is, so even N = 64 runs fine
 * (the naive "one thread per tile" design would need 64*64 = 4096
 * threads per block and exceed the 1024-thread hardware limit).
 *
 * The () operator also measures pure kernel time with CUDA Events and
 * returns it in milliseconds, so the timing comparison can be printed.
 */

#ifndef SRC_PIXELFILTER_HPP_
#define SRC_PIXELFILTER_HPP_

#include "ImageFilter.hpp"

class PixelFilter : public ImageFilter {
protected:
	/// edge length of one mosaic tile, in pixels (e.g. 8, 16, 32, 64)
	unsigned int tileSize;
public:
	//                       depthIn, depthOut, tile edge length N
	PixelFilter(unsigned int dIn, unsigned int dOut, unsigned int n) :
		ImageFilter(dIn, dOut), tileSize(n) {};

	/// change the mosaic tile size between runs (used by the benchmark)
	void setTileSize(unsigned int n) { this->tileSize = n; }
	unsigned int getTileSize() const { return this->tileSize; }

	/// the mosaic grid uses a fixed 16x16 CUDA block, independent of N
	virtual void resizeGrid(unsigned int w, unsigned int h) {
		this->threads = dim3(16, 16);
		this->grid = dim3((w + this->threads.x - 1) / this->threads.x,
						   (h + this->threads.y - 1) / this->threads.y);
	}

	/// run the filter; returns the measured kernel time in milliseconds
	float operator()(const unsigned char *input, unsigned char *output,
					 const unsigned int w, const unsigned int h);
};

#endif /* SRC_PIXELFILTER_HPP_ */
