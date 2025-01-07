from sentence_transformers import SentenceTransformer,util
import torch 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('paraphrase-distilroberta-base-v1', device=device)

# Two lists of sentences
sentences1 = ['The cat sits outside',
             'A man is playing guitar',
             'The new movie is awesome'] ## 最相似

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great'] ## 最相似

# 获得词向量
embedings1 = model.encode(sentences1, convert_to_tensor=True)
embedings2 = model.encode(sentences2, convert_to_tensor=True)

# 计算余弦相似度
cosine_scores = util.pytorch_cos_sim(embedings1, embedings2)

# 输出结果
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
