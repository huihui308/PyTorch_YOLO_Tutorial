
# RTCdet的介绍

## 网络结构
RTCDet的网络结构近似于我在这个项目里实现的YOLOv7的结构，核心模块是ELANBlock，可以直接看代码，不难理解；Neck用的就是SPP模块，PaFPN用的是YOLOv7风格的PaFPN，但把通道数适当扩大了2倍，并且针对PaFPN重新设计了一个ELANBlockFPN，整个PaFPN结构要比v7的厚一些，最后就是YOLOX风格的解耦检测头。
最后，设计了width和depth因子，来设计不同规模的RTCDet检测器；

## 训练配置
这一块适当参考了MMYOLO的RTMDet，使用AdamW优化器，weight decay=0.5，baselr=0.001（对应的batch size为64），根据batch size的大小自动调整学习率；使用了Model EMA；Cosine学习衰减策略；数据增强使用的是YOLOX风格的：




