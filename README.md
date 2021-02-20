## Lung Diseases 🚑
<hr>
<br>

<br>

<!-- PROJECT LOGO -->

<h3 align="center">Deep learning applied healthcare</h3>

<p align="center">
 An awesome AI case!
 <br />
 <br />
    <br />
  </p>
</p>


<br>
<br>





### Description 📃

A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. In our current release, there are 1200 COVID-19 positive images, 1341 normal images, and 1345 viral pneumonia images. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients, this project has been used with a business case in Gavb consulting, that is a AI consulting company in Brazil, thus the project was done in portuguese.


<br>
<br>


### ResNet50 architecture 

 We using to ResNet50 architecture as a model for this case, we were thought about this architecture applied a medical image replacing some layers and make a fine tuning in this model. We've put a new layers and some made changes in the architecture by the way when come up needs in our project and fiting for the final architecture for solution.
 
 <br>
 <br>
 
 ![alt text](https://i.stack.imgur.com/gI4zT.png)
 
 
 <br>
 <br>
 <br>
 
 
 
 
 
 ### AI application 📱
 
 It's a gif that shows the final application done in this project, a web app created by streamlit framework who be able the doctor sees the real diagnostic x-ray of pacient, it's application would will be running in the cloud provider.
 
<br>
 
 
 
 
 ![Alt Text](https://github.com/felipeoliverai/lung_diseases-cv/blob/main/utils/examples/2021-02-20_09-33-01_example_1.gif)

<br>
<br>


The application it's userful for a doctor, that can see another landscapes of the x-ray, because the AI help him to identify another disease which doctor can't saw, 
the some steps for until inference are: 

 
 * 1: Input x-ray image at the app, right now application will be running inference for under background, consistent in a cnn called Resnet50 make prediction on the image.
 * 2: click in "Predição" in english called out "Prediction" for make prediction of image.
 * 3: the under bottom will show called "GradCAM" this button is responsible about interpretability of the prediction, it's tell us what was seen by the AI and why your choice was this. 
 
 

 
 
<br>
<br>


 
 
 
 ### Prerequisites
 
 <br>
1. Get the API key 
2. Clone the 
  ```sh 
  $ git clone https://github.com/felipeoliverai/lung_diseases-cv.git
  ```
 

2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```


 
 


