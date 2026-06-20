/*
 * EffectFilter.hpp
 *
 * CUDA port of the OpenCL EffectFilter (Parallel Worlds 3).
 *
 * The effect filter takes TWO input images:
 *   1. the original COLOR image (3 bytes/pixel)         -> dInput
 *   2. the grey EDGE image from the Sobel filter (1 b/p) -> dEdge
 * and writes a COLOR image (3 bytes/pixel)               -> dOutput.
 *
 * Wherever the edge value at a pixel is above `threshold`,
 * the matching colour pixel is darkened (each channel * 0.5).
 * Everywhere else the colour is copied through unchanged.
 */

#ifndef SRC_EFFECTFILTER_HPP_
#define SRC_EFFECTFILTER_HPP_

#include "ImageFilter.hpp"

/// This class applies a sobel-edge-based darkening effect to a colour image.
// It needs its own edge buffer on the GPU in addition to the input/output
// buffers it inherits from ImageFilter, so it overrides resizeBuffers().
class EffectFilter : public ImageFilter {
protected:
	/// pointer to the grey EDGE image on the GPU
	unsigned char *dEdge;
	/// colour depth of the edge image in bytes (1 for greyscale)
	unsigned int depthEdge;
public:
	EffectFilter(unsigned int dIn, unsigned int dEdge, unsigned int dOut) :
		ImageFilter(dIn, dOut), dEdge(nullptr), depthEdge(dEdge) {};

	virtual ~EffectFilter() {
		// dInput/dOutput are freed by the base class destructor;
		// we only own dEdge here.
		SAFE_CALL(cudaFree(dEdge));
	}

	/// in addition to the base buffers, (re)allocate the edge buffer
	virtual void resizeBuffers(unsigned int currWidth, unsigned int currHeight) {
		unsigned int bytesEdge = currWidth * currHeight * depthEdge;
		// grow the edge buffer only when the image got bigger;
		// note: we test BEFORE the base class updates width/height
		if (currWidth * currHeight > width * height) {
			SAFE_CALL(cudaFree(dEdge));
			SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dEdge), bytesEdge));
		}
		// let the base class (re)allocate dInput/dOutput and update width/height
		ImageFilter::resizeBuffers(currWidth, currHeight);
	}

	/// run the effect: colour `input` + grey `edgeInput` -> colour `output`
	void operator()(const unsigned char *input, const unsigned char *edgeInput,
					unsigned char *output,
					const unsigned int w, const unsigned int h,
					const float threshold);
};

#endif /* SRC_EFFECTFILTER_HPP_ */
