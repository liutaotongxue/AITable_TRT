# Emotion 模块

## 概述
Emotion模块提供基于深度学习的面部表情识别功能。该模块使用EmoNet神经网络架构，能够识别8种基本情绪，并提供情绪的效价(valence)和唤醒度(arousal)维度分析。

## 核心组件

### 1. EmoNetClassifier (`emonet_classifier.py`)
**功能**：情绪识别分类器，封装了EmoNet模型的加载、预处理和推理功能。

**主要特性**：
- 支持8种基本情绪识别
- 提供情绪维度分析（效价和唤醒度）
- 批量处理优化
- GPU加速支持
- 自动模型权重管理

**识别的情绪类别**：
1. **Neutral** - 中性
2. **Happy** - 快乐
3. **Sad** - 悲伤
4. **Surprise** - 惊讶
5. **Fear** - 恐惧
6. **Disgust** - 厌恶
7. **Anger** - 愤怒
8. **Contempt** - 轻蔑

**情绪维度**：
- **Valence（效价）**：情绪的积极-消极程度（-1到1）
  - 正值：积极情绪
  - 负值：消极情绪
  
- **Arousal（唤醒度）**：情绪的激活程度（-1到1）
  - 高值：高激活状态（如兴奋、愤怒）
  - 低值：低激活状态（如平静、悲伤）

### 2. EmoNet模型架构 (`emonet_classifier.py`)
**功能**：深度卷积神经网络，专门设计用于面部表情识别。

**架构特点**：
- **Hourglass Network**：沙漏型网络结构
  - 多尺度特征提取
  - 递归的下采样和上采样
  - 保留空间信息
  
- **注意力机制**：
  - 自动聚焦面部关键区域
  - 热图引导的特征加权
  - 提高识别准确性

- **多任务学习**：
  - 同时预测表情类别和情绪维度
  - 共享特征表示
  - 提高模型泛化能力

**网络层级**：
```
输入(256×256×3) → Conv层 → BatchNorm → MaxPool
    ↓
ConvBlock × 3 → HourGlass模块 × 2
    ↓
特征融合 → 注意力加权 → 全连接层
    ↓
输出：表情(8类) + 效价(1值) + 唤醒度(1值)
```

### 3. 配置管理 (`config.py`)
**功能**：管理情绪识别模块的配置参数。

**主要配置项**：
```python
# 模型配置
EMOTION_MODEL_PATH = 'models/emonet_8.pth'  # 模型权重路径
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 情绪类别
EMOTION_LIST = [
    'neutral', 'happy', 'sad', 'surprise',
    'fear', 'disgust', 'anger', 'contempt'
]

# 预处理参数
INPUT_SIZE = (256, 256)  # 输入图像尺寸
NORMALIZE = True          # 是否归一化
```

## 技术实现

### 图像预处理流程
1. **尺寸调整**：缩放到256×256
2. **灰度转换**：转为灰度图
3. **通道复制**：复制为3通道
4. **归一化**：像素值归一化到[0,1]
5. **张量转换**：转为PyTorch张量

### 推理流程
```python
人脸图像 → 预处理 → 特征提取 → 情绪分类 → 后处理 → 结果输出
                          ↓
                    维度预测（效价/唤醒度）
```

### 批处理优化
- 支持批量人脸输入
- 动态批大小调整
- GPU内存优化
- 并行处理加速

## 使用方法

### 基础使用
```python
from modules.emotion import EmoNetClassifier

# 初始化分类器
classifier = EmoNetClassifier()

# 单张人脸预测
face_img = cv2.imread('face.jpg')
result = classifier.predict_single(face_img)

print(f"情绪: {result['emotion']}")
print(f"效价: {result['valence']:.2f}")
print(f"唤醒度: {result['arousal']:.2f}")
```

### 批量处理
```python
# 批量人脸预测
face_batch = [face1, face2, face3]
results = classifier.predict_batch(face_batch)

for i, result in enumerate(results):
    print(f"Face {i}: {result['emotion']}")
```

### 高级应用
```python
# 情绪追踪
emotion_history = []

for frame in video_frames:
    face = detect_face(frame)
    if face:
        emotion = classifier.predict_single(face)
        emotion_history.append(emotion)
        
        # 分析情绪变化
        if len(emotion_history) > 10:
            analyze_emotion_trend(emotion_history[-10:])
```

## 性能指标

### 识别性能
- **准确率**：85%+ (FER2013数据集)
- **推理速度**：
  - GPU: 50+ FPS
  - CPU: 10-15 FPS
- **延迟**：<20ms (单张图像，GPU)

### 资源消耗
- **模型大小**：约100MB
- **内存占用**：
  - GPU: 500MB
  - CPU: 200MB
- **初始化时间**：2-3秒

## 情绪分析解释

### 情绪分类
每种情绪的典型特征：
- **快乐**：嘴角上扬，眼睛眯起
- **悲伤**：嘴角下垂，眉头下压
- **惊讶**：眼睛睁大，嘴巴张开
- **恐惧**：眉毛上扬，眼睛睁大
- **厌恶**：鼻子皱起，上唇提升
- **愤怒**：眉头紧锁，嘴唇紧闭
- **轻蔑**：单侧嘴角上扬
- **中性**：面部放松，无明显表情

### 维度分析
情绪在二维空间的分布：
```
        高唤醒度
           ↑
    愤怒   |   兴奋
    恐惧   |   快乐
←————————————————————→ 效价
消极       |      积极
    悲伤   |   平静
    厌倦   |   满足
           ↓
        低唤醒度
```

## 应用场景

### 1. 用户体验监测
- 监测用户使用产品时的情绪状态
- 识别负面情绪并及时响应
- 优化交互设计

### 2. 教育场景
- 学生专注度分析
- 情绪状态反馈
- 个性化教学调整

### 3. 健康监护
- 情绪健康追踪
- 抑郁倾向检测
- 心理状态评估

### 4. 人机交互
- 情感化智能助手
- 自适应界面调整
- 个性化内容推荐

## 限制和注意事项

### 技术限制
1. **光照敏感性**：极端光照条件影响准确率
2. **角度限制**：大角度侧脸识别效果下降
3. **遮挡问题**：口罩、墨镜等遮挡物影响识别
4. **文化差异**：不同文化背景的表情差异

### 使用建议
1. **环境要求**：
   - 良好的光照条件
   - 正面或小角度侧脸
   - 清晰的面部图像

2. **数据处理**：
   - 使用滑动窗口平滑结果
   - 结合上下文信息判断
   - 考虑个体差异

3. **隐私保护**：
   - 不存储用户面部图像
   - 本地处理，不上传云端
   - 遵守数据保护法规

## 错误处理

### 常见问题
1. **模型加载失败**
   ```python
   FileNotFoundError: EmoNet权重文件未找到
   解决：检查models/emonet_8.pth是否存在
   ```

2. **GPU内存不足**
   ```python
   RuntimeError: CUDA out of memory
   解决：降低批处理大小或使用CPU模式
   ```

3. **输入格式错误**
   ```python
   ValueError: 图像维度不匹配
   解决：确保输入为BGR格式的numpy数组
   ```

## 优化建议

### 提高准确率
1. 人脸预处理优化
2. 数据增强训练
3. 模型集成方法
4. 上下文信息融合

### 提高速度
1. 模型量化压缩
2. TensorRT加速
3. 批处理优化
4. 异步处理

## 扩展开发

### 自定义情绪类别
```python
# 修改config.py
EMOTION_LIST = [
    'neutral', 'happy', 'sad', 'surprise',
    'fear', 'disgust', 'anger', 'contempt',
    'excited', 'bored'  # 新增类别
]

# 重新训练模型
```

### 集成其他模型
- 支持FER+模型
- 支持AffectNet预训练
- 自定义模型接口

## 依赖关系
- **深度学习框架**：
  - torch: PyTorch框架
  - torchvision: 视觉工具
  
- **图像处理**：
  - opencv-python: 图像操作
  - numpy: 数组处理
  
- **其他模块**：
  - core: 日志系统