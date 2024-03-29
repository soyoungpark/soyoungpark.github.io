---
title: "Security Checklist for Developing a Basic Web Application"
layout: post
date: 2017-09-18 18:52
image: /assets/images/nodejs.png
headerImage: false
hidden: true # don't count this post in blog pagination
tag: 
- security 
- webdev
star: false
category: blog
author: soyoungpark
description: Security checklist for web development beginners
---

This is a basic security checklist for creating a basic web application. Apt for students and developers who want to create a basic Node.js web application. It is assumed that readers already know how to build or have already built a functioning web application. 

---

## Motivation

This summer (June - Aug 2017), I had the opportunity to teach a group of high school students about making a simple web application (two of them chose NodeJS, the other three PHP). Excited at seeing their applicaitons come to life, the students were tempted to finish off their project with an application vulnerable to common security attacks, such as SQL injection, XSS, and dictionary (or other variants, like rainbow table) attacks on the password. This, to beginners in web development, is understandably an appealing thought, since implementing new features and changing user interface may seem like more 'tangible' progress than coding functions to prevent security breaches. However, if the web application is to be made public -- which was the case for all of my students -- it is crucial that the application adheres to basic but important security principles. Following is a list of some very basic security measures to implement, what attacks they help prevent, example(s) if applicable, and why each measure is important to web security. 

---

## The Checklist

### 1. Hash your password, with salt

{% highlight js %}

/* when user creates a new account, inputs password ... */

var crypto = require('crypto'); // cryptography package for password encryption

var sha512 = function(password, salt){
    var hash = crypto.createHmac('sha512', salt); /** SHA-512 hashing algorithm */
    hash.update(password);
    var value = hash.digest('hex');
    return {
        salt:salt,
        passwordHash:value
    };
};

/* ... */

var usersalt = genRandomString(16);
var passwordData = sha512(password, usersalt);
var hashed_pw = passwordData.passwordHash;

/* save to our database both the usersalt (bound to be unique for each user account) and the hashed_pw */

{% endhighlight %}

### 2. Prevent SQL injection:

3.
4.
5.
6.



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