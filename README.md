# pinkteam

imageprocessing.py

turbidity.m

# Turbidity Detection of Sample Images using the Principles of Scattering and Image Processing

## Introduction-

This project intends to give an alternative of expensive and sophisticated conventional Turbidity Meters. In this, I have used the principle of Scattering and Image Processing to
meausre the Turbidity of a Water Sample.
Exploiting the fact that whenever light collides with a solid particle, it scatters maximum at 90 degree angle, Image Processing empowers to find the concentration of the solid particles in the water sample, and thus gives the turbidity.

## How to use the Software?

1- Clone the repository or download the ZIP file
2- Unzip the file and run Turbidity.m file
3- Select the image sample from the given sample images
![Capture](https://user-images.githubusercontent.com/53121012/119776213-44bc2780-bee2-11eb-8d3b-d6c0e40024cb.PNG)
4- Wait for the process to complete. A message box will pop up displaying the Turbidity value.
Sample Image used-
![Capture1](https://user-images.githubusercontent.com/53121012/119776341-6ddcb800-bee2-11eb-9077-78878f29b397.PNG)
Output-
![Capture](https://user-images.githubusercontent.com/53121012/119776351-703f1200-bee2-11eb-9277-847f7b6160d1.PNG)

turbidity_gpt.py
The Underwater Image Quality Measure (UIQM) mentioned in REAL-TIME ENHANCEMENT OF VISUAL CLARITY IN TURBID WATERS FOR COMMERCIAL DIVERS AND ROVS is an objective metric designed to evaluate the visual quality of underwater images. UIQM provides a numerical score assessing the image's information content, specifically focusing on three key aspects:

Color Faithfulness: This component evaluates how closely the colors in the image match expected natural colors, which is crucial since underwater images often suffer from color distortions due to light absorption and scattering.
Shape Clarity: This measures the distinctness of edges and shapes, which tend to blur or fade in underwater scenes.
Contrast Quality: Higher contrast enhances visibility and detail, which is generally diminished in underwater images.
By integrating these aspects, UIQM offers a single score reflecting the image's quality, useful for comparison between processed (enhanced) and raw images to quantify the improvement in visibility.

Explanation of the Code
Colorfulness Calculation: This metric is based on the deviation between red-green and yellow-blue channels, following the principle that more vibrant images have greater color differences.
Contrast Calculation: Here, we use the standard deviation of pixel values, assuming that higher variance indicates better contrast.
Sharpness Calculation: The Laplacian variance method helps quantify edge clarity, as high edge variance indicates more discernible shapes.
This script gives a UIQM score by combining these components, aligning with the purpose of UIQM as described in REAL-TIME ENHANCEMENT OF VISUAL CLARITY IN TURBID WATERS FOR COMMERCIAL DIVERS AND ROVS.
