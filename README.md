### Introduction

Self driving and navigation is and has been a goal for many individuals and companies for quite a few years now. From autonomous cars to robots that can navigate our world there are so many tasks that have consumed too many man hours. The fun of this challenge is it is still unsolved. Autonomous robots path planning can still be improved, self driving cars are only recently sharing the roads with us in very limited fashions. Many of the limitations have been in hardware. There are still many hardware limitations but as hardware improves we can push our algorithms and models.
 
2004 sparked the start of DARPA’s Grand Challenge. While no car completed the challenge it was a good start to initiating the research into self driving cars. Hardware at the time was very limited. In DARPA’s final report it lists the vehicles hardware ranging from the meager laser rangefinders and ultrasonic sensors to the expensive LiDARs. Not much about the hardware is stated but Graphics cards at the time were not being used yet for neural networks [Using GPUs for Machine Learning Algorithms, 2005].
 
 While today LiDARs are becoming more financially attainable, good ones are still too expensive for mass production and they are still a bit bulky. While they may be large and relatively expensive they are still currently the go to option for self driving companies such as Waymo and Chevy. Waymo is at the forefront of self driving and proudly sitting on top of their vehicles is a massive LiDAR. Waymo for the most part likes to stay under the radar. ![picture of waymo van](/assets/waymo_van.png)

Tesla is very well known for pushing self driving into the hands of average consumers. Tesla is able to achieve this by keeping costs down and using a mix of cheaper sensors. Tesla currently uses an array of mostly ordinary cameras all around the vehicle. Historically Tesla had also used radar but they cut it out of their cheaper models when they were hard to source during and following the years after Covid 19. Tesla or maybe more accurately Elon Musk believes full self driving is possible using cameras only since humans are able to drive with only our eyes. The higher end models still sport a radar and there have been rumors of Tesla bringing them back to their lower end models. Time will tell if Tesla is forced to give up their ambitions of full self-driving without LiDAR or similar sensors.
![picture of teslas cameras](/assets/tesla_sensors.jpg)

Sensor mixing and processing is still an ongoing challenge for anyone attempting to produce an autonomous system. It is not trivial to train nor conduct inference when there is so much data. Processing and moving around so much data is not an easy task and there are still many hurdles to overcome with modern hardware. Similar issues come up again when trying to use these trained models since these autonomous systems are moving and need to make decisions in very short amounts of time.

Each sensor can be better and worse at different tasks. A camera can be good at detecting signs, signals, people, other cars, etc. A lidar is very good at detecting depth over all 360 degrees. There are stereo cameras for detecting depth in a cone in front of a sensor as well. These stereo cameras are not as good as a lidar but they are much cheaper. 
This project was an exploration of trying to create a simple neural network to drive a car in the game BeamNG.drive. BeamNG.drive started out as more of a crashing simulator but evolved into a fairly good driving and racing simulator. As stated before there are many challenges in creating the full system for this sort of problem.

### Data gathering and prep

Data for this project was all generated manually. Originally using a game controller was the plan. There are very few libraries for capturing and playing back gamepad inputs. Capturing the gamepad inputs was only successful for the joystick of the controller. The inputs from the controller were sometimes very good but at times the data was a bit chaotic. The intent was to get input data that looked like an inverted v, as seen in the graph. Since there was doubt in the data the best option was to record a small sample of data and play that data back. Data playback was where using a gamepad fell apart. Upon searching there was only one ancient python library and drivers that could do this. Upon testing the library and drivers it became apparent that this method would not work. The drivers were very unreliable and they broke gamepad support on the system being used for data gathering.
![Attempted trigger input](/assets/trigger_graph.png)

Once gamepad use was ditched keyboard input was the next option. Python has a good keyboard library that makes getting and playing back keyboard data relatively easy. There were two options for gathering keyboard input. First was grouping the keys together. As an example “a+w” would become one input “w” would be another input. Total there would be 8 or 11 with 8 being the better choice as it’s very unlikely that the user would input  combinations above two keys. The input  “a+s+d” is most likely a mistake. With this approach we would need to ensure that the model has as many outputs as chosen options. The model could then use one hot encoding and choose one option at a time.
 
Instead the option was chosen to encode the inputs into a binary array. Each index in the array would represent a key. This choice is also more flexible because if other keys were added there is less work that needs to be done to add the combinations to the code and dataset. Only w, a, s, and d keys were accepted as inputs. The inputs were recorded into a csv file. It is worth noting that it could be a good idea to also add the corresponding frame name to the row in the csv file. 

For capturing the frames of the game the first idea was to use Nvidias shadowplay software. Nvidia Shadowplay is a software intended to be incorporated easily onto games. Shadowplay can hook onto any games running modern game APIs such as vulkan. Originally BeamNG.drive uses DirectX11 but Shadowplay was not hooking in. Luckily the game also supports vulkan and Shadowplay was able to hook into it. 
Game capture and keyboard capture were now both working. After tying them together a problem was discovered. The number of frames and the number of inputs did not match up. For a shorter recording they may be off by eight and a longer one was off by 12. There was too much concern about poor quality data so this method was abandoned. 

Capturing the frames of the game in a timely manner and matching them to the current input was needed. There were a few libraries that were tested out. Some that used built in Windows libraries and others that took screenshots from the specified portion of the frame. Dxcam was the chosen option. Dxcam is a library on github that’s intent was to be as fast as possible. Getting smooth consistent frames is important for training. When using data in a time series fashion the data needs to be predictable and not be skewed by bad data gathering. Using dxcam allowed the frames to be captured alongside the keyboard inputs at a consistent 30 frames per second. 
Saving the images was still an issue. It was fast enough to capture the frames and store them in memory but when each frame takes up 2.4MB system memory filled up too quickly to finish a race. Fortunately solid state drives have become cheap and readily available to average consumers. Looking around at some old tech forums the average write speeds for an HDD were 70MBs to 150MBs. That would not leave much room for anything else at the 72MBps the system was capturing at. Frames were being captured at 720p to save on storage space and computation. There was a trade off of lower resolution to save on space and allow for storing more data. If more storage and compute was available 1080p video capture would be better. If more storage was available then it would be better to capture at a higher resolution and scale down the images for training and inference but that was not the case. Online storage is fairly cheap but the problem would then be transferring the data. The dataset for this project ended up being around 250GB and it would be better if there was much more data. Every lap was around 2200 frames give or take depending on how fast the car was driven around the track. With 113,000 frames that is only a meager 51 laps. While the data was captured at 720p the data was kept in RGB format. Keeping the color allows filters to be applied as wanted. When saving they had to be saved in .tif format. Too much data is lost to compression when saving the .jpeg format. .png is very common but saving .png files was too slow to maintain 30fps capture while .tif format was able to save quickly enough to maintain 30fps. 30fps was chosen as the theory was it would give enough time to conduct inference and it would give the model enough information when training. 30fps gives 33.33ms to capture, modify, and infer on the frame. When path planning for a robot the pathing algorithm is normally allowed to keep running as many updates as it can within a given time frame. The faster all of the components are the more time the algorithm has to plan out a path. Since the pathing will be done by a neural network inference as well as the image processing needs to happen within a reasonable time. The goal was to try and maintain 30fps. 

### Analysis

The planned algorithms for this problem are a CNN and a CNN with some form of temporal or time series attached. Inspiration for time series was from Ramin Hasani who in an interview stated “The real world is all about sequences. Even our perception — you’re not perceiving images, you’re perceiving sequences of images,” he says. “So, time series data actually create our reality.” [“Liquid” machine-learning system adapts to changing conditions, 2021]
All models were built using Tensorflow Keras. Starting out with a basic CNN the model had two Conv2D layers with a MaxPooling2D layer after each convolutional layer. The convolutional layer is used to recognize patterns in the images they are fed. Recognizing these patterns in a video should give us our spatial information. A Conv2D layer has arguments for number of filters and kernel size. A convolutional layer will apply a convolution or more correctly a Cross-Correlation to a window as we slide the window across an image. The cross correlation Kernel is different for each filter. The intent for filters is to be able to detect different patterns or properties of the image. If we are detecting cats one filter may be very good for detecting eyes and another the ears. Together all of the filters can help the model identify the goal prediction. 
The goal for the Conv2D layer in the context of driving can be a lot. For driving on the course selected in BeamNG.drive we’re interested in edge detection for detecting the edges of the road as well as our distance from them. Since the speed is displayed on the frame as well there is a possibility that speed could be learned as well. There are no other cars or signals, or pedestrians to worry about. BeamNG.drive is not an arcade game so the driving dynamics are more challenging. If more obstacles and challenges were introduced the model would need more data to train on which would require more computation.
![beamng picture](/assests/beamng.png)

The next layer is a MaxPooling2D. A max pooling layer is good to downscale the image to make it easier to process. The important bits of the image are kept from a max pooling layer since our previous Conv2D layer picked out those important bits. Max pooling can also help the model as the important objects in the image shift, rotate, and scale. Since we are downscaling the important bits with max pooling, their scale in the original image is less important. For rotating the filter should still be able to pick out its features up to a limit. Finally for shift the filter is picking out the feature regardless of where it is in the image.
Two of the models got dense layers just before their output layer. These dense layers are the more typical neural network consisting of many nodes with weighted connections between them. General dense layer networks are fairly capable in solving many problems so it was added to the network to increase its learning potential.
 
For temporal information a LSTM layer was chosen. TimeDistributed was also attempted but with the format of the data it was challenging to attach the timesteps to the data for training. Since the data is streamed in a video format there is temporal or time series information that the model can learn. The vehicle should be moving but if we only look at individual frames it can be hard to discern movement. The intent of the LSTM layer is for the model to better decide on a current decision depending on the previous frame, current frame, and possibly guessing at what can occur in the future depending on the decision it makes. The LSTM will learn temporal information and can give better decision making on what to do next. Another model consisted of many ConvLSTM2D layers. These layers are similar to a LSTM except the image transformations are convolutional.

The models themselves sizes were kept small so that they and a decent amount of training data could fit within VRAM. With Tensorflow that can be challenging because Tensorflow allocates 90% of VRAM by default. Most GPU monitoring software is only able to report the amount of VRAM allocated instead of the amount being utilized. Below is how nvtop reports vram utilization.
![nvtop vram utilization](/assets/nvtop.png)

Not reporting out VRAM does make pushing the limits of the system challenging. Using a Nvidia 3090 has 24GB of VRAM which is a lot but it’s also not. Optimizing around the available hardware is a must to get in as many epochs as possible. Analyzing the time taken for the GPU to process the data loaded into system memory it took both around eight seconds per batch of about 256 images. In order to get more epochs in loading the data into system memory with the CPU it would be best to load data into system memory while the GPU is processing what it has in VRAM. One issue with python is that its garbage collection can be slow. There were issues with garbage collection taking too long to start itself so it had to be kicked off manually after clearing out the previous batch of images. 

### Results  

The CNN summary is
Model: "sequential" 
_________________________________________________________________  
Layer (type)            	Output Shape          	Param #  
_________________________________________________________________  
conv2d (Conv2D)         	(None, 718, 1278, 32) 	896  
max_pooling2d (MaxPooling2D  (None, 359, 639, 32) 	0  
)  
conv2d_1 (Conv2D)       	(None, 356, 636, 32)  	16416  
max_pooling2d_1 (MaxPooling  (None, 178, 318, 32) 	0  
2D)  
flatten (Flatten)       	(None, 1811328)       	0  
dense (Dense)           	(None, 32)            	57962528  
dense_1 (Dense)         	(None, 4)             	132  
_________________________________________________________________  
Total params: 57,979,972  
Trainable params: 57,979,972  
Non-trainable params: 0

8/8 - 14s - loss: 0.5364 - accuracy: 0.8516 - 14s/epoch - 2s/step  
Test accuracy: 85.16%  
8/8 - 13s - loss: 1.0107 - accuracy: 0.3945 - 13s/epoch - 2s/step  
Test accuracy: 39.45%  
8/8 - 13s - loss: 0.6145 - accuracy: 0.8086 - 13s/epoch - 2s/step  
Test accuracy: 80.86%  
8/8 - 13s - loss: 0.3536 - accuracy: 0.8633 - 13s/epoch - 2s/step  
Test accuracy: 86.33%  
8/8 - 13s - loss: 0.8767 - accuracy: 0.5312 - 13s/epoch - 2s/step  
Test accuracy: 53.12%  
8/8 - 13s - loss: 0.6910 - accuracy: 0.5195 - 13s/epoch - 2s/step  
Test accuracy: 51.95%  
8/8 - 13s - loss: 0.4089 - accuracy: 0.8008 - 13s/epoch - 2s/step  
Test accuracy: 80.08%  
1/1 - 1s - loss: 1.3149 - accuracy: 0.0667 - 904ms/epoch - 904ms/step  
Test accuracy: 6.67%  
total accuracy: 0.6045247400179505  
Removing the outlier of 6.67% since it’s not using a full test set brings the total to 68.14%

While running the game Tensorflow reports 96ms/step   and setting the time with python reports inferent time 0.1248 seconds. Using the logic used in robotics path planning we could run two inference cycles on the image and average the two together within our 33.33ms window. Ideally we would be able to run inference even faster and average even more decisions together. With two inference runs together it pushed to a max of 0.2636ms. That does not leave much time for grabbing the outputs but with output turned off the program should run marginally quicker and keep within the 33.33ms. 

Running the same model but with the images converted to greyscale:

8/8 - 13s - loss: 2.2993 - accuracy: 1.0000 - 13s/epoch - 2s/step  
Test accuracy: 100.00%  
8/8 - 13s - loss: 10.0264 - accuracy: 0.5859 - 13s/epoch - 2s/step  
Test accuracy: 58.59%  
8/8 - 13s - loss: 2.7213 - accuracy: 0.8281 - 13s/epoch - 2s/step  
Test accuracy: 82.81%                                                                                                                                                                           8/8 - 13s - loss: 1.1277 - accuracy: 0.8555 - 13s/epoch - 2s/step  
Test accuracy: 85.55%  
8/8 - 13s - loss: 9.8338 - accuracy: 0.5859 - 13s/epoch - 2s/step  
Test accuracy: 58.59%  
8/8 - 13s - loss: 3.2228 - accuracy: 0.6484 - 13s/epoch - 2s/step
Test accuracy: 64.84%
8/8 - 13s - loss: 2.1499 - accuracy: 0.7930 - 13s/epoch - 2s/step  
Test accuracy: 79.30%  
1/1 - 1s - loss: 18.9156 - accuracy: 1.0000 - 845ms/epoch - 845ms/step  
Test accuracy: 100.00%  
total accuracy: 0.787109375

Greyscale was chosen in hopes to lower inference time. This idea turned out to be wrong. To lower inference time either better hardware needs to be chosen or the dimensionality of the model needs to be reduced with pruning or dropping weights.Inference time lowered marginally to 0.1185ms. What reducing the image from RGB to grayscale did do was lower the amount of training time required. Under the otherwise same settings one batch took eight seconds for RGB vs one second for greyscale. Our first run with RGB was able to do eight epochs in 3 hours. Our second model was ran for 32 epochs. In game the model did much better. It gets close to being able to make a turn. The model turns too soon but if we help it out by getting it to the turn manually it almost makes the complete turn.

The CNN with LSTM summary is:  
Model: "sequential"  
_________________________________________________________________  
Layer (type)            	Output Shape          	Param #  
_________________________________________________________________  
conv2d (Conv2D)         	(None, 718, 1278, 32) 	320  
max_pooling2d (MaxPooling2D  (None, 359, 639, 32) 	0  
)  
time_distributed (TimeDistr  (None, 359, 20448)   	0  
TimeDistributed)  
lstm (LSTM)             	(None, 16)            	1309760  
flatten_1 (Flatten)     	(None, 16)            	0  
dense (Dense)           	(None, 32)            	544  
dense_1 (Dense)         	(None, 4)             	132  
_________________________________________________________________  
Total params: 1,310,756  
Trainable params: 1,310,756  
Non-trainable params: 0


8/8 - 12s - loss: 0.3284 - accuracy: 1.0000 - 12s/epoch - 2s/step  
Test accuracy: 100.00%  
8/8 - 12s - loss: 0.5135 - accuracy: 0.5781 - 12s/epoch - 1s/step  
Test accuracy: 57.81%  
8/8 - 12s - loss: 0.4547 - accuracy: 0.7578 - 12s/epoch - 1s/step  
Test accuracy: 75.78%  
8/8 - 12s - loss: 0.2470 - accuracy: 1.0000 - 12s/epoch - 1s/step  
Test accuracy: 100.00%  
8/8 - 12s - loss: 0.4533 - accuracy: 0.8125 - 12s/epoch - 1s/step  
Test accuracy: 81.25%  
8/8 - 12s - loss: 0.3930 - accuracy: 0.8633 - 12s/epoch - 1s/step  
Test accuracy: 86.33%  
8/8 - 12s - loss: 0.2822 - accuracy: 0.9258 - 12s/epoch - 1s/step  
Test accuracy: 92.58%  
1/1 - 1s - loss: 0.6492 - accuracy: 1.0000 - 999ms/epoch - 999ms/step  
Test accuracy: 100.00%  
total accuracy: 0.8671875  

The LSTM model after 25 epochs was able to achieve a 86.72% accuracy which was promising. But this is where testing on images only fails. The model does not work and only predicts a constant “w” and “a” during a live run. Every prediction is [[0.82755864 0.516616   0.06827126 0.04048353]]. It is possible the number of parameters is too low for this application. From the summary the model has 2.3% as many parameters. A larger model could be constructed but then it takes up more memory during training. The demetioinality for the dense hidden layer goes down signifiantly compared to the CNN model. It is possible the very large dense of the CNN is doing the heavy lifting and it is why it's out performing the LSTM model.

### Conclusion

Almost all computer science problems can be broken down into the transfer and manipulation of data. This is especially true when it comes to training a model to drive a car. As predicted the storage, loading, and processing of the data was a challenge. 250GB of data seems like a lot but in reality for training a model to drive it is very little. Over 32 epochs eight terabytes of data will move through the model. Moving around so much data needs to be done in an efficient manner. Tensorflow has some data loading libraries to help but the data format needs to match. Since tif images are not supported it would have been useful to convert the images to a supported format. Ideally the loading and processing of the images would be done in parallel between the CPU and the GPU. Loading the data took around as long as training the data took. Tensorflow does allow some of the pre-processing to be done on the GPU which can also speed things along. Hopefully with time libraries like direct storage will come to tensorflow and the system memory can be bypassed.

Validating the data can be challenging when attempting to solve issues that can be solved in many ways without any single one being completely correct. While there may be optimal solutions they are not necessarily needed. Currently companies are training their models with real world and simulation data. To effectively test driving models it is best to have a simulation software and validation system set up. It’s even more important when driving around a one to two ton vehicle. One side effect of training with real world data is that it can skew the results. In the early days of Tesla self driving software it was common to hear people say it drives like a californian driver. Their cars were also reportedly struggling more outside of typical United States infrastructure until they started gathering more data outside of the United States. This project suffered from not having enough training data and no simulation to test in. While the validation dataset would see results in the 90s and averaging at 80% for the CNN and LSTM combined model. The model did not work as it only produced a single output. The models had to be tested live manually. The test dataset was not robust enough to get a good idea if the model was actually working.

The application of the models was very important. For a moving dataset getting the temporal data trained into the model correctly is key. The model needs to know where it has been and what actions it has taken to be able to make better decisions. Being able to predict where it will go with predictions is also key. The datasets need to be modified to know how long the sequence is otherwise the model assumes each batch is the sequence length. Since only 32 frames or less were able to fit into VRAM that is only one second of time. One second of time is not very much time when traveling at high speeds.

Time series information is also important, otherwise the model may try and learn when to turn after a given amount of time instead of why to turn. CNN with a low number of epochs could not learn when to turn at the correct time. The model would drive the car straight for some time and veer off left too early. It is possible with more data and more epochs that the CNN would better learn when to turn. The training data was mostly consistent with the vehicle starting at the same spot, going the same direction and attempting to enter turns at a set speed. It is quite possible that having the data consistent In this manner did not allow the model to learn why to turn instead of it seemingly trying to guess when time wise to turn.

More epochs and more data are needed. The models struggled with sections of the dataset. The accuracy between epochs would be high for a majority of the time and low for the others. It would be good to save off and analize the bad sections of data so that more targeted training data could be generated. Since it takes a long time to generate data knowing the shortcomings of the dataset would allow time to be better spent. 
Software compatibility is important. Windows is not compatible with newer versions of tensorflow GPU. Only being able to run inference on the CPU was detrimental to understanding if the model was working well. Inference took 92ms to run on the CPU. At 92ms the model is running at 11fps. Going from color to grayscale did not allow for faster inference. The model would need to be simplified or pruned to allow for faster inference. What would be best is running inference of the GPU or other more capable hardware. AI accelerators are becoming more available being built into CPUs and systems on a chip, SoCs.





