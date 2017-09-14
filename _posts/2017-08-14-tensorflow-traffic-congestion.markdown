---
title: "Tensorflow Traffic Congestion Classifier"
layout: post
date: 2017-08-14 22:10
tag: tensorflow 
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

From the very beginning, I knew I wanted to merge my two majors — Urban Studies and Computer Science — for my independent research. I also knew I wanted to do this by using dataset produced by the city. Because of the broadness of the term, ‘urban data’ could include anything from census data and pollution statistics to crime rate. While reading a blog post from Sidewalk Labs, however, I was motivated by the idea of applying computer vision to traffic camera feeds. 

What was a problem that could benefit from application of computer vision to traffic camera video footage? I initially thought of a system that sent real-time alert when an accident occurred. I had thought that my machine learning model will be trained to see and identify vehicles on the otherwise fixated screen (the model will of course weed out pedestrians), and log an alert message whenever two or more of the vehicles are within a few centimeters away from each other. A few weeks into Coursera Machine Learning course (taught by Andrew Ng), however, I realized that such a task would be a lot more complicated than I had envisioned, and quickly revised the project goal to logging alert messages in the event of heavy traffic congestion. It will still be very useful for the city residents, and is a more apt task for image recognition through supervised learning. My model had the advantage of not needing to install additional hardware (sensors) throughout intersections.

### Approach

I had intended to source the traffic video footage from NYC Department of Transportation (http://dotsignals.org/), which opened up to the public its real-time traffic camera feeds. For some network issues, however, I was not able to see the stream, although the browser’s network panel clearly showed a stream of .ts files being asynchronously received. I quickly switched the source to EarthCam (https://www.earthcam.com/), which offered HD live camera feeds from 120 cameras installed in New York City, and hundreds more from around the world. 

For the first four weeks of the project, I took a Machine Learning class on Coursera, taught by Andrew Ng, professor at Stanford and former head of Baidu AI / Google Brain. I completed all the homework tasks and updated the progress on my blog (www.soyoung-park.blogspot.kr). I also spent about a week finalizing my tool for the project. I had learned the basics of supervised learning, but which toolset was the most appropriate for the project? After looking into the pros and cons of Keras, Caffe, Scikit Learn, and Tensorflow, I chose Tensorflow, mainly because it is more low-level and helpful for when I want to implement deep learning algorithms myself. For this particular project, as it is my first project using a machine learning library, I would mainly be using Tensorflow’s high-level library, but I was excited by the possibility of switching to low-level later on in my career for custom implementation of deep learning algorithms. 

As for training the model, because of the sheer size of input data that I was dealing with (video data), I learned that it could potentially take up to weeks to fully train the model from scratch. For the purpose of this research, however, I needed to be able to experiment and quickly pivot the direction if necessary. So I decided to re-train the final layer of Inception v3, which is an image classificaiton model already trained on extensive image data (ImageNet). The model was not specialized as a traffic congestion classifier, however, so I needed to fine-tune the model by training its last layer.

### Description of Experiment & Analysis

#### Phase 1:

I had initially collected images of what I think are 'heavily congested' traffic stuation vs those that are less congested, by manually searching Google images. My rationale for doing so was that while it may be time-consuming, I would be able to provide a diverse set of ‘heavy traffic congestion’ images, taken from various angles, capturing different scenarios for traffic congestion. I had then wanted to create a universal traffic congestion classifier model which, when given an image of any highway or traffic intersection in the world, would tell whether it is heavily congested or not. I had also thought this approach (manual Google-searching) would clearly prevent the Tensorflow model from using the same data for testing and training, which, according to Coursera Machine Learning course, was to be avoided at all costs.


![Markdowm Image][6]
<!--![Markdowm Image][/images/tensorflowtraffic/heavytraffic2.jpg]-->
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


[Check it out](http://soyoungpark.github.io/tftraffic/) here.
