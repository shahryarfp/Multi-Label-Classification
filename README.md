# Multi-Label-Classification
ML and DL approaches to predict flood in an urban area

### Summary

In Houston, TX, we are facing a challenge with flooding in an urban area. To tackle this issue, we are looking to develop a machine learning model capable of predicting the outcome of flood events based on their initial conditions. We have a collection of 3,000 simulated flood incidents (training set), each serving as a historical data point to train our model.

### Dataset

Here you can see a sample image from the area(it's an elevation map of area):

<!-- ![sample image from dataset](./readme_images/sample.jpg) -->
<img src="./readme_images/sample.jpg" width="300" height="300">

You can find a complete dataset for this project here:
[Dataset](https://drive.google.com/drive/folders/1LSgzWgiDrNdlXfBmZFl1LyrFVWFaUA_q?usp=share_link)

### Explanation
Complete Explanation of project could be found in the file, "Project Description.pdf"
The project consists of two parts:
1. Segmentation task
2. Diagnosis Task

#### Segmentation task:
The dataset for this task is available here:
[Dataset](https://drive.google.com/drive/folders/1LSgzWgiDrNdlXfBmZFl1LyrFVWFaUA_q?usp=share_link)

We have three types of masks:
1. ground glass
2. consolidation
3. pleural effusion

In order to do the segmentation task, three U-Net models are trained, and they are available here:
[Trained Models](https://drive.google.com/drive/folders/1ubOYddgXB_DkUQwLnlASKzLqA0vo4P1q?usp=share_link)

#### Diagnosis task:
The dataset for this task is available here:
[Dataset](https://drive.google.com/drive/folders/1ubOYddgXB_DkUQwLnlASKzLqA0vo4P1q?usp=share_link)

In order to do the diagnosis task, a VGG-16 model is used and trained. It is available here:
[Trained Model](https://drive.google.com/drive/folders/1ubOYddgXB_DkUQwLnlASKzLqA0vo4P1q?usp=share_link)

#### How to use:
1. Just simply open the code
2. Correct the links to the dataset and models
3. Run the code

#### Results

Results of the segmentation task:
<!-- ![seg task result](./readme_images/mask-result.png) -->
<img src="./readme_images/mask-result.png" width="300" height="600">

Result of the diagnosis task:

<!-- ![dia figure](./readme_images/figure.png) -->
<img src="./readme_images/figure.png" width="500" height="600">

It reached 92% accuracy for the test data.
Final results:
<!-- ![Final result](./readme_images/final.png) -->
<img src="./readme_images/final.png" width="500" height="500">


### Reference

Yao, Hy., Wan, Wg. & Li, X. A deep adversarial model for segmentation-assisted COVID-19 diagnosis using CT images. EURASIP J. Adv. Signal Process. 2022, 10 (2022). https://doi.org/10.1186/s13634-022-00842-x

