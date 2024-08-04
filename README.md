# Algorithm-Interview
1.题目一：
    主要利用图的构建和深度优先搜索（DFS）来计算最大爆炸数目


    
2.题目二：
    1）实现过程
       （1）按段落存储PDF文件
       （2）使用 SentenceTransformer 模型将提取的段落转换为嵌入向量
       （3）使用 Faiss 创建索引
    2）最终结果：
    Based on the context, the following tricks are used when training PP-YOLO:
    
    1. DropBlock: This is a trick that replaces the original DropBlock with a new branch that adds additional computational cost but improves the performance of the detector.
    
    2. Grid Sensitive: This is a trick that uses a grid-sensitive approach to improve the performance of the detector.
    
    3. Matrix NMS: This is a trick that uses a matrix NMS to improve the performance of the detector.
    
    4. CoordConv: This is a trick that uses a CoordConv layer to improve the performance of the detector.
    
    5. Better Pretrain Model Using a pretrain model with higher classiﬁcation accuracy on ImageNet may result in better detection performance. Here we use the distilled ResNet50-vd-dcn[13] as the backbone networks unless speciﬁed. The architecture of FPN and head in our basic models is completely the same as YOLOv3[32]. The details have been presented in section 3.1. We initialize in this paper. Since there are already a lot of works to study backbone network and to explore data augmentation, we do not repeat them in this paper. Searching for hyperparame-
    ters using NAS often consumes more computing power, so there is usually no condition to use NAS to perform a hyperparameter search in each new scenario. Therefore, we still use the manually set parameters following YOLOv3[32].
    
    6. SPP: This is a trick that integrates SPM into CNN and uses max-pooling operation instead of bag-of-word op-
    eration.
    
    7. CoordConv: This is a trick that views the SPP as a convolutional layer and uses max-pooling instead of bag-of-word op-
    eration.
    
    8. Better Pretrain Model Using a pretrain model with higher classiﬁcation accuracy on ImageNet may result in better detection performance. Here we use the distilled ResNet50-vd-dcn[13] as the pretrain model[29] . This obviously does not affect the efﬁciency of the detector.
    
    9. Matrix NMS: This is a trick that uses a matrix NMS to improve the performance of the detector.
    
    10. SPP: This is a trick that integrates SPM into CNN and uses max-pooling operation instead of bag-of-word op-
    eration.
    
    11. CoordConv: This is a trick that views the SPP as a convolutional layer
           
