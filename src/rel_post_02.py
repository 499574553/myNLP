# encoding=utf-8

import os
import sys
import jieba
import uniout

import scipy as sp

from sklearn.feature_extraction.text import TfidfVectorizer

seg_list1 = jieba.cut("这是一篇关于机器学习的文章，实际上它没有多少有趣的东西。")
str1 = " ".join(seg_list1)
print("分词结果：")
print(str1)

seg_list2 = jieba.cut("图像数据库会变得非常巨大。")
str2 = " ".join(seg_list2)
print(str2)

seg_list3 = jieba.cut("大多数图像数据库可以永久存储图像。")
str3 = " ".join(seg_list3)
print(str3)

seg_list4 = jieba.cut("图像数据库可以存储图像。")
str4 = " ".join(seg_list4)
print(str4)

seg_list5 = jieba.cut("图像数据库可以存储图像。图像数据库可以存储图像。图像数据库可以存储图像。")
str5 = " ".join(seg_list5)
print(str5)

seg_list=[str1, str2, str3, str4, str5]

stpwrdlst = ['多少', '它', '可以']

vectorizer = TfidfVectorizer(min_df=1, stop_words = stpwrdlst)
print("停用词：")
print(vectorizer.get_stop_words())
X = vectorizer.fit_transform(seg_list)
b2=vectorizer.get_feature_names()
print("词语序列：")
print(b2)
print("TF IDF Vector：")
print(X.toarray())

print("测试文章：")
new_post = jieba.cut("图像数据库。")
str_new = " ".join(new_post)
print("测试文章分词结果：")
print str_new
new_post_vec = vectorizer.transform([str_new])

print("测试文章的向量表示：")
print(new_post_vec.toarray())


def dist_norm(v1, v2):
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())

#print X.getrow(0);

print("相似度计算结果：")

similar1 = dist_norm(X.getrow(0), new_post_vec)
print 'post1: ' + str(similar1)

similar2 = dist_norm(X.getrow(1), new_post_vec)
print 'post2: ' + str(similar2)

similar3 = dist_norm(X.getrow(2), new_post_vec)
print 'post3: ' + str(similar3)

similar4 = dist_norm(X.getrow(3), new_post_vec)
print 'post4: ' + str(similar4)

similar5 = dist_norm(X.getrow(4), new_post_vec)
print 'post5: ' + str(similar5)

