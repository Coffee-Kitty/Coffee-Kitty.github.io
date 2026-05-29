

# VLLM

https://www.bilibili.com/video/BV1kx4y1x7bu/?spm_id_from=333.337.search-card.all.click&vd_source=39767bfbc4ae772d0c2f8d8b32b54ce6



## KV Cache

![image-20241222170724769](/assets/legacy-picture/image-20241222170724769.png)

![image-20241222165606637](/assets/legacy-picture/image-20241222165606637.png)



kv cache存在的问题

![image-20241222170757529](/assets/legacy-picture/image-20241222170757529.png)

vllm对此做了优化：

优化效果如下图：

![image-20241222170903166](/assets/legacy-picture/image-20241222170903166.png)

## 优化一： Page Attention



![image-20241222165928075](/assets/legacy-picture/image-20241222165928075.png)





![image-20241222171051456](/assets/legacy-picture/image-20241222171051456.png)





![image-20241222171217974](/assets/legacy-picture/image-20241222171217974.png)





![image-20241222171244458](/assets/legacy-picture/image-20241222171244458.png)





## 优化二：sharing KV Blocks

当需要模型对同一个prompt生成多个 response时，

![image-20241222171400787](/assets/legacy-picture/image-20241222171400787.png)



![image-20241222171601702](/assets/legacy-picture/image-20241222171601702.png)



![image-20241222171611890](/assets/legacy-picture/image-20241222171611890.png)

## 代码调用

![image-20241222171651497](/assets/legacy-picture/image-20241222171651497.png)



## 在vllm上自定义新模型

https://www.bilibili.com/video/BV1xbypYtEu2/?spm_id_from=333.337.search-card.all.click&vd_source=39767bfbc4ae772d0c2f8d8b32b54ce6