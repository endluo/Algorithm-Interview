# Algorithm-Interview

大模型RAG流程：

    1）实现过程
       （1）按段落存储PDF文件,通过size大小只保存正文
       （2）使用 SentenceTransformer 模型将提取的段落转换为嵌入向量
       （3）使用 Faiss 创建索引，使用重新排序搜索
       
    2）最终结果：
    
        Based on the given context, some of the tricks that can be used when training PP-YOLO are:
        
        1. Convolutional Neural Networks (CNNs)
        2. Batch Normalization (BN)
        3. ReLU activation function
        4. Dropout regularization
        5. Learning rate scheduling
        6. Early stopping
        7. Optimizers such as Adam and RMSprop
        8. Data augmentation techniques such as random cropping, flipping, and rotation
        9. Transfer learning from pre-trained models
        10. Hyperparameter tuning
        
        These are just a few examples of the types of tricks that can be used when training PP-YOLO. However, it's important to note that the specific tricks used will depend on the specific task and dataset being trained on. Additionally, the effectiveness of these tricks may vary depending on various factors such as the size of the dataset, the complexity of the model architecture, and the quality of the training data.
        
        In summary, there are several tricks that can be used when training PP-YOLO, including convolutional neural networks (CNNs), batch normalization, ReLU activation function, dropout regularization, learning rate scheduling, early stopping, optimizers such as Adam and RMSprop, data augmentation techniques, transfer learning from pre-trained models, hyperparameter tuning, and more.
           
