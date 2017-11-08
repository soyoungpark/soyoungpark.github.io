---
title: "Tensorflow Traffic Congestion Classifier"
layout: post
date: 2017-08-14 22:10
tag: 
- tensorflow 
- transportation
- computervision
image: /assets/images/tensorflowtraffic/tf.png
headerImage: true
projects: true
hidden: true # don't count this post in blog pagination
description: "NYC 5th Avenue traffic congestion real-time classifier using Tensorflow"
category: project
author: soyoungpark
externalLink: false
---

Trained a Tensorflow model, fine-tuning last layer of Inception v3, to build a traffic congestion level classifier. 

---

### Tech stack

- Python
- Tensorflow
- Docker
- bash

---

### Motivation

From the very beginning, I knew I wanted to merge my two majors — Urban Studies and Computer Science — for my independent research. I also knew I wanted to do this by using dataset produced by the city. Because of the broadness of the term, ‘urban data’ could include anything from census data and pollution statistics to crime rate. While reading a <a href="https://medium.com/sidewalk-talk/sidewalk-held-a-two-day-midsummer-engineering-hackathon-heres-what-we-built-45df0e061eda#5daf">blog post</a> from Sidewalk Labs, however, I was motivated by the idea of applying computer vision to traffic camera feeds. 

What was a problem that could benefit from application of computer vision to traffic camera video footage? I initially thought of a system that sent real-time alert when an accident occurred. I had thought that my machine learning model will be trained to see and identify vehicles on the otherwise fixated screen (the model will of course weed out pedestrians), and log an alert message whenever two or more of the vehicles are within a few centimeters away from each other. A few weeks into Coursera Machine Learning course (taught by Andrew Ng), however, I realized that such a task would be a lot more complicated than I had envisioned, and quickly revised the project goal to logging alert messages in the event of heavy traffic congestion. It will still be very useful for the city residents, and is a more apt task for image recognition through supervised learning. My model had the advantage of not needing to install additional hardware (sensors) throughout intersections.

### Approach

I had intended to source the traffic video footage from NYC Department of Transportation (http://dotsignals.org/), which opened up to the public its real-time traffic camera feeds. For some network issues, however, I was not able to see the stream, although the browser’s network panel clearly showed a stream of .ts files being asynchronously received. I quickly switched the source to EarthCam (https://www.earthcam.com/), which offered HD live camera feeds from 120 cameras installed in New York City, and hundreds more from around the world. 

For the first four weeks of the project, I took a Machine Learning class on Coursera, taught by Andrew Ng, professor at Stanford and former head of Baidu AI / Google Brain. I completed all the homework tasks and updated the progress on my blog (www.soyoung-park.blogspot.kr). I also spent about a week finalizing my tool for the project. I had learned the basics of supervised learning, but which toolset was the most appropriate for the project? After looking into the pros and cons of Keras, Caffe, Scikit Learn, and Tensorflow, I chose Tensorflow, mainly because it is more low-level and helpful for when I want to implement deep learning algorithms myself. For this particular project, as it is my first project using a machine learning library, I would mainly be using Tensorflow’s high-level library, but I was excited by the possibility of switching to low-level later on in my career for custom implementation of deep learning algorithms. 

As for training the model, because of the sheer size of input data that I was dealing with (video data), I learned that it could potentially take up to weeks to fully train the model from scratch. For the purpose of this research, however, I needed to be able to experiment and quickly pivot the direction if necessary. So I decided to re-train the final layer of Inception v3, which is an image classificaiton model already trained on extensive image data (ImageNet). The model was not specialized as a traffic congestion classifier, however, so I needed to fine-tune the model by training its last layer.

### Description of Experiment & Analysis

#### Phase 1:

I had initially collected images of what I think are 'heavily congested' traffic stuation vs those that are less congested, by manually searching Google images. My rationale for doing so was that while it may be time-consuming, I would be able to provide a diverse set of ‘heavy traffic congestion’ images, taken from various angles, capturing different scenarios for traffic congestion. I had then wanted to create a universal traffic congestion classifier model which, when given an image of any highway or traffic intersection in the world, would tell whether it is heavily congested or not. I had also thought this approach (manual Google-searching) would clearly prevent the Tensorflow model from using the same data for testing and training, which, according to Coursera Machine Learning course, was to be avoided at all costs.

<img src='../assets/images/tensorflowtraffic/heavytraffic1.jpg' />
<img src='../assets/images/tensorflowtraffic/heavytraffic2.jpg' />
<figcaption class="caption">Photos of heavy traffic congestion I found through Google image search</figcaption>


After spending half a day searching for traffic photos on Google images, I realized this method is not only extremely time-consuming (a lot more so than I had estimated) but also ineffective for the task at hand. It was very difficult to find images of heavily congested roads and intersections. After a few solid hours searching, I ended up with just 67 images of traffic congestion, but very few images of roads that are less congested. Most importantly, many of the ‘heavy congestion’ images were either very high resolution and/or taken from angles that traffic camera could not be positioned at.

#### Data Collection, Phase 2:

I then quickly changed my data collection strategy to snapshotting the EarthCam live feed every three minutes. I wrote and ran a bash script that screen-shots and saves the traffic camera stream as a time-stamped jpg file in a specified directory. 

{% highlight js %}
while [ 1 ];
do vardate=$(TZ=America/New_York date +%d\-%m\-%Y\_%H.%M.%S); 
screencapture -t jpg -x -R 241,136,957,539 ~/Dev/tensorflow_practice/inceptionv3/trafficphotos/captured/$vardate.jpg; 
sleep 180; 
done
{% endhighlight %}

By running this script at different intervals, I collected about 400 images throughout different times of the day, at different days of the week. I was finally ready to test with Tensorflow. 

In using Tensorflow I mostly followed Google Codelab, and Tensorflow for Poets. As per instruction, I installed docker, which is analogous to virtual machine but share the host OS kernel (and usually the binaries and libraries too) and are much lighter. Running Tensorflow within Docker allows for easier migration in the future and, as I learned through trial and error, is much easier than running it in my native OS X. In docker image, I provided two image directories, each named after my desired classifications: 'heavily-congested', and 'less-congested'. I then ran Tensorflow with 500 training steps. 

{% highlight python %}
python retrain.py \
  --bottleneck_dir=bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=inception \
  --summaries_dir=training_summaries/basic \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=capturedtraffic
{% endhighlight %}

I wanted to create a binary classifier which, when given a snippet from traffic camera, outputs the probability of the image containing heavy traffic congestion and sends out real-time alert (HTTP POST request) to a predefined API endpoint. This binary classification (‘heavy congestion’ vs ‘else’) suffices for the purpose of this research because NYC residents are likely to be more interested in ‘heavy congestion’ events than in any of the ‘else’ events (ranging from zero traffic to light traffic).

The actual training took about half an hour at this stage. Throughout the training, I was able to see that the Tensorflow was validating the new data against the current model at every training step. Naturally, the validation score increased as more data was fed into the model. After the training was over, I could run ‘label_image.py’ on any test image to output the percentage of the photo representing ‘heavy congestion’ or ‘else’. The results seemed to make sense, at the least. The model predicted that the photo below on the left was 58.3% likely to be less heavy congestion (currently labelled as ‘else’, then as ‘nojam’), and predicted that the photo below on the right was 97.3% likely to be ’heavy congestion’ (which then was labelled as ‘jam’ — I later relabelled the two categories to make them more intuitive).




While 58.3% was a better performance than just randomly guessing out of the two possible categories, I noticed something strange. The model predicted that some images of empty roads were ‘heavy congestion’, with more than 80% conviction (score above 0.80). Clearly something was wrong. Tackling this issue was initially not very straight-forward because I used Tensorflow at a very high level and did not control a lot of its environment, other than some parameters when training the model. This meant that either the task I wanted to accomplish (traffic level classifier) was not appropriate for Tensorflow, or I was making some mistake in setting the parameters, or I was providing not enough (or not diverse enough) photos for training data. 

After talking to Professor Murphy who advised me on this research, I realized that while my traffic camera is fixated, it regularly makes rotations, streaming NYC 5th Avenue from different angles. This meant that I may need to provide ample (~300) data, for each angle, for each category. 5 angles x 300 x 2 categories meant that I needed about 3000 photos total. Also, for my training data I also needed traffic photos of the 5th Avenue in different weather conditions, and photos snapshotted under poor network condition (blurry or low resolution). Considering these new requirements, I decided that taking snapshots was no longer a valid strategy for collecting data because it was early July, and I doubted that it would snow in NYC in summer. 

#### Data collection, phase 3:

After browsing through the EarthCam website, I found that for every traffic camera, there exists an archive of images from at least half a year ago (for I saw images from March). I decided to write a Python script that scrapes all the images in the archive. I initially had some difficulties because the archive was loaded through asynchronous javascript, which shot HTTP GET request to the EarthCam server endpoint (https://www.earthcam.com/cams/common/gethofitems.php). I simulated this GET request at a predefined interval to mimic AJAX GET request, and succeeded in seamlessly scraping images from the server. 






In less than a few hours, I was able to collect about 6000 images, taken from February 2017 to August 4, 2017. I eliminated about 1200 irrelevant images before sorting them. Following are the types of images that were opted out from sorting: images that are too blurry, dark, or low-resolution even for human eyes to determine what’s going on, images from when the roads are entirely blocked out for parades or protests, and images snapshotted from when the traffic camera was rotating. My rationale behind excluding the said images was that if a pair of human eyes cannot categorize them into either heavy traffic or else, then the model trained through supervised learning should not be expected to categorize them either. Shown below are examples of such images.

Again, I had to manually sort them by angles (ranging from angle 0 to 4), then into one of the two traffic categories. I ran the experiment twice: first, by 10 categories (heavycongestion0, lesscongestion0, heavycongestion1, lesscongestion1, … heavycongestion4, lesscongestion4), second, by just two (heavycongestion, lesscongestion). 

I also launched Tensorboard to monitor the training process; I was able to monitor real-time updates to the validation accuracy and cross entropy, which were also logged to the terminal console. Cross entropy is a way of measuring how far away from the expected answer (classification) the current model is; a model that predicts correctly with 60% probability (‘score’ in Tensroflow) is more distant from an ideal model than a model that predicts correctly with 90% probability. I specified 500 training steps for each experiment, and not surprisingly, training accuracy increased and cross entropy decreased with each step for both experiments. I would like to discuss the pros and cons of the two approaches below.

#### Experiment 1 (Sorted by angle and traffic level): 10 categories

<img src='../assets/images/tensorflowtraffic/accuracy1.png' />

Orange line represents train accuracy, and the turquoise line validation accuracy. Tensorflow partitions the input data into training set and validation set. The idea is analogous to crossfold validation in traditional statistics, bur more appropriate for machine learning tasks which usually take in a larger size of dataset. The training set, which constitutes majority of the input data, is used to train the model by adjusting parameters and fit weights. Validation set is used during the training to measure the improvement in the model’s accuracy as more training steps are taken. While both the training and validation accuracy increase over time, and though the final test accuracy is  80.6%, there is a considerable gap overall between the training and validation throughout all 500 steps.  

<img src='../assets/images/tensorflowtraffic/crossentropy1.png' />

Cross entropy is measured for both the training and validation dataset. Both decrease over time, meaning the model has become more sophisticated over time in correctly predicting the output. 

#### Experiment 2 (Sorted by traffic level alone): 2 categories

<img src='../assets/images/tensorflowtraffic/accuracy2.png' />

It is said that the gap between train and validation accuracy often comes from overfitting. While both train and validation accuracy increased as more training steps were taken, the gap started to widen from about 300th training step. Final test accuracy was 85.7%. 

<img src='../assets/images/tensorflowtraffic/crossentropy2.png' />

Not surprisingly, cross entropy decrease over time (training step). 

I tested this model with 28 images, representing different angles and different level of congestion. What was interesting was that because my input dataset for training included only images of both extremes (extremely heavy congestion or empty road), the score for each prediction often matched the congestion ‘level’ of the test image. For instance, for the following image of sparsely populated, but not completely empty intersection, the score for ‘heavy congestion’ was 0.69670. 




For test images belonging to the extremes, the accuracy was high. 0.98415 score for light congestion for the following photo demonstrates that the model does not simply interpret bright lights as heavy traffic. 



The model also understood that congestion level is not necessarily proportional to how much the road is exposed.


For unrelated images, the results were also satisfactory. It was close to random guessing. 


I observed a few discrepancies, too. The model classified the photo below to be light congestion rather than heavy congestion. 


It was a satisfactory and efficiently built model, overall, but I decided to run it again with the same input data but with less training steps. This was because I noticed the gap between train and validation accuracy after the 300th step, and thought that completing the the model training before it started overfitting might help solve some of the discrepancy I observed. 


#### Experiment 2, take two (500 -> 300 training steps)

Final test accuracy was 83.3%. Although it is lower than the test accuracy from when 500 training steps were taken, the drop may be due to prevention of overfitting. As can be seen in the graph below, there is no sign of overfitting, as far as the difference between the train and validation accuracy is concerned. 

But when I ran the test on the same image which had previously been classified in an unexpected manner, the score for ‘light congestion’ has decreased but the prediction itself remained the same: light congestion. I wish I could take images that were incorrectly classified and find some commonality among them, but there were only two such images in the test dataset of 28 images (this dataset is completely separate from the input dataset). I was, however, satisfied to train a model with less overfitting. 

### Conclusion

In Neural Information Processing Systems 2015, D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, and Todd Phillips presented a paper discussing the technical debt incurring from applying software engineering framework to real world machine learning problems. In the paper, the five scholars state that “only a tiny fraction of the code in many ML systems is actually devoted to learning or prediction.” 

I had indeed spent a good portion of the time sourcing, saving and sorting data. Because this was my first time applying deep learning to a real world problem, I was able to learn new concepts (Coursera Machine Learning) and apply it with a toolset that I have never used before (Tensorflow run on Docker image). Applying theory through Coursera Machine Learning course homework was one thing, but applying it to real-world data, which was not always in the format I wanted, and often contained unexpected type of data (the protests and parades were entirely unexpected), was another.

In the future I would like to start other machine learning projects which train the model from scratch — for more sophisticated fine-tuning. Although they are more time consuming, I believe I would be able to better customize the powerful multithreading power of Tensorflow to solve city problems by using city-produced data. 



[Check it out](http://soyoungpark.github.io/tftraffic/) here.




[1]: http://www.soyoungpark.github.io/assets/images/tensorflowtraffic/heavytraffic1.jpg
[2]: http://www.soyoungpark.online/assets/images/tensorflowtraffic/heavytraffic2.jpg
[3]: ../assets/images/tensorflowtraffic/heavytraffic2.jpg
[6]: http://kune.fr/wp-content/uploads/2013/10/ghost-blog.jpg