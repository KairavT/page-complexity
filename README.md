# Page Complexity
Estimates how visually complex a page of text is from a photo, using OpenCV. It scans the photo into a clean page, measures the layout, and returns a single complexity score.

## Results
The pipeline takes a photo of a page, flattens it, and scores it from three layout measurements. On a set of test photos it correctly ranks a denser, tighter page above a more spaced out one.

## How it works
The photo goes through two stages.

First it is turned into a clean scan. The image is blurred and run through Canny edge detection, the edges are dilated to close gaps, and the largest contour is taken as the page. Its four corners are found and a perspective transform flattens the page into a straight on rectangle. Adaptive thresholding then turns it into clean black text on white paper.

Then the flattened page is measured. Three things are computed: text density (the fraction of the page that is ink), line spacing (the median gap between lines of text), and text size (the median height of each line). Line spacing and text size come from a horizontal projection profile, which counts the black pixels in each row so that text lines show up as peaks and the gaps between them show up as valleys.

The three measurements are normalized and combined into one score, where denser text, tighter spacing, and smaller text all push the score higher.

## Limitations
The score is sensitive to photo quality, not just the page. The same page photographed sharply and then blurry gave noticeably different scores, because blur thins the text and widens the measured gaps. So this works best as a relative tool comparing pages photographed under similar conditions, not as an absolute measure.

The pipeline also needs a clear photo of a single page on a plain, high contrast background. A cluttered background breaks the page detection, and the code raises an error when it cannot find four clean corners.

The normalization ranges used in the score are estimated by hand rather than fitted from data.

## Tech Stack
- Python
- OpenCV
- NumPy
- matplotlib

## How to Run
```bash
git clone https://github.com/KairavT/page-complexity.git
cd page-complexity
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy matplotlib
python3 main.py
```
Place photos of pages in the `images/` folder and update the list of paths at the bottom of `main.py`.