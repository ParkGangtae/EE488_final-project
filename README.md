20190235 Gangtae Park

**EE488 Final Project Report**

1. **Key Design Choices**

**Scheduler**

I applied cosine annealing LR scheduler from pytorch. It repeats from initial learning rate, 0.001 to 0.0001, which is eta\_min and follows cosine function. It can prevent the model to stuck in local minimum. The higher learning rate help model to escape local minimum. On the other hand, lower learning rate tries to make model in good accuracy. Eventually, the model should work in good accuracy. That is the reason why I set maximum epoch to odd times of period of cosine function.

**Filtering Noise**

We couldn’t ignore the noises of dataset of train2. Therefore, I had to filter the noises after I finish training with train1. As same as practical 5, I calculated the cosine similarity from average of embeddings of each label. I generated a new noise-free dataset by filtering out the 15 percent with the low cosine similarity. This procedure doesn’t need many epochs, so I set the max\_epoch as 15 and period of cosine annealing LR schedule as 5 and loss function is simple softmax function.

**Data Augmentation**

Dataset of train1 includes Caucasian male, besides dataset of train2 includes Korean male and female face. Basically, they have a difference in tone of the skin color. Therefore, I used color jitter function and set hue as 0.5. Plus, there were numbers of images that include sunglasses or glasses. Thus, I added sunglasses image to 20 percent of dataset with function, add\_sunglasses. Moreover, to give a variation for each data, I applied random flip, rotation, and grayscale.

**Loss Function**

According to the paper [2], ArcFace loss is more fit to create the boundary between each class. So, I adopted ArcFace loss function instead of softmax or triplet loss. The recommended scaling factor was 64. Also, I used margin as 0.2 instead of recommended number 0.5 because I have more classes than the author had. After I trained with ArcFace loss function, I tried to combine softmax and ArcFace because softmax focus on labeling the data to which class it belongs to, but ArcFace focus on creating boundary. Therefore, I thought that it would make a synergy.

2. **Ablation Study**

The results of ablation study are in [Table 2-1]

||<p>**Cosine**</p><p>**Annealing LR**</p>|**Filtering**|<p>**Data**</p><p>**Augmentation**</p>|**ArcFace**|**Combined**|**ViT**|**Accuracy**|**EER**|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|[1]|✗|✗|✗|✗|✗|✗|86\.45%|13\.55%|
|[2]|✓|✗|✗|✗|✗|✗|87\.87%|12\.14%|
|[3]|✓|✓|✗|✗|✗|✗|89\.13%|10\.87%|
|[4]|✓|✓|✓|✗|✗|✗|91\.22%|8\.78%|
|[5]|✓|✓|✓|✓|✗|✗|91\.32%|8\.68%|
|[6]|✓|✓|✓|✗|✓|✗|92\.43%|7\.57%|

[Table 2-1]



The hyperparameters of each model are in [Table 2-2]

||**batch\_size**|**test\_interval**|**max\_epoch**|
| :-: | :-: | :-: | :-: |
|[1]|100|5 / 5|50 / 35|
|[2]|128|5 / 5|50 / 50|
|[3]|128|5 / 5|50 / 50|
|[4]|128|10 / 5|50 / 15|
|[5]|128|10 / 5|50 / 15|
|[6]|128|10 / 5|50 / 15|

[Table 2-2]

I tried to match the batch size as the power of 2. By the multiple times of experiment, early stopping the finetuning for train set 2 at about 15<sup>th</sup> epoch performs the best for validation dataset.

3. **Conclusion**

In this project, the model's performance improved from an initial accuracy of 86.45% and EER of 13.55% to a final accuracy of 92.43% and EER of 7.57%. Plus, the test set showed EER of 8.15%. Efforts were made to avoid focusing on superficial features such as skin color, hair, and facial hair, ensuring the model relied on more meaningful features. Although ResNet-18 was used, attempts to explore alternative models were limited by parameter constraints. Combining the two existing loss functions showed promise, but the lack of new loss function ideas remains a missed opportunity for further enhancement.

In summary, the model's performance has improved significantly, but there are still opportunities for optimization, particularly in exploring novel loss functions and expanding the model architecture within available resources.

4. **Reference**

[1] I. Loshchilov and F. Hutter, “SGDR: Stochastic Gradient Descent with Warm Restarts”, 2017.

[2] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition,", 2018.

