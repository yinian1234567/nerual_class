from transformers import AutoModel, AutoFeatureExtractor

# 加载模型和特征提取器
model_name = "facebook/dinov2-base"
model = AutoModel.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# 处理输入图像
from PIL import Image
image = Image.open(r"D:\File\PycharmProject\NeuroScience\images\0\cat.png")
inputs = feature_extractor(images=image, return_tensors="pt")

# 获取特征
outputs = model(**inputs)
features = outputs.last_hidden_state
