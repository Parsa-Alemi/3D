# 3D
Machine Learning Project for Converting 2D and 3D images
<br>
<b>This Project was created in 2019 </b>

# Purpose
In this project, I aimed for converting 2D images with only 1 sample and angle to 3D images because we have a lot of old data in the world which we don't have several pictures from them to create a 3D model out of them so I  was aiming for solving this problem with Deep Learning.

# Adversarial Nets
I used A generative adversarial network <a href="https://en.wikipedia.org/wiki/Generative_adversarial_network">(GAN)</a> in this project Before this I tried using a lot of other solutions to this problem but in the end, I came to the understanding that this is the best solution. I already put some sample models from my training in the <a href="https://github.com/Parsa-Alemi/3D/tree/master/Models">Models Folder</a> so you can access them.
# Dataset
Dataset was created by putting 2 cameras side by side and getting each photo labeled, the Network is trying to guess the second picture from the dataset to have 2 pictures of 2 different angles so we can try and create a 3D Model. 
## Loss functions
<img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Ftowardsdatascience.com%2Funderstanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a&psig=AOvVaw3AVGX-QSIISyRwt1JgECJQ&ust=1631418748272000&source=images&cd=vfe&ved=0CAkQjRxqFwoTCODX6sqC9vICFQAAAAAdAAAAABAD" alt="Loss Functions" width="500" height="600">


#### you can download a sample npy file in <a href="https://drive.google.com/file/d/1GH0drmCPXmtvdzJIXkPHLfNZPeYYKIX7/view?usp=sharing">Here</a>
