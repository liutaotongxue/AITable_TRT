"""
情绪识别分类器模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
from . import config
from ..core.logger import logger


nn.InstanceNorm2d = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.InstanceNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.InstanceNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.InstanceNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest') 

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class EmoNet(nn.Module):
    def __init__(self, num_modules=2, n_expression=8, n_reg=2, n_blocks=4, attention=True, temporal_smoothing=False):
        super(EmoNet, self).__init__()
        self.num_modules = num_modules
        self.n_expression = n_expression
        self.n_reg = n_reg
        self.attention = attention
        self.temporal_smoothing = temporal_smoothing
        self.init_smoothing = False

        if self.temporal_smoothing:
            self.n_temporal_states = 5
            self.init_smoothing = True
            self.temporal_weights = torch.Tensor([0.1,0.1,0.15,0.25,0.4]).unsqueeze(0).unsqueeze(2).cuda() #Size (1,5,1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.InstanceNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.InstanceNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            68, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(68,
                                                                 256, kernel_size=1, stride=1, padding=0))

        for p in self.parameters():
            p.requires_grad = False


        if self.attention:
            n_in_features = 256*(num_modules+1) #Heatmap is applied hence no need to have it
        else:
            n_in_features = 256*(num_modules+1)+68 #68 for the heatmap
        
        n_features = [(256, 256)]*(n_blocks)

        self.emo_convs = []
        self.conv1x1_input_emo_2 =nn.Conv2d(n_in_features, 256, kernel_size=1, stride=1, padding=0)
        for in_f, out_f in n_features:
            self.emo_convs.append(ConvBlock(in_f, out_f))
            self.emo_convs.append(nn.MaxPool2d(2,2))
        self.emo_net_2 = nn.Sequential(*self.emo_convs)
        self.avg_pool_2 = nn.AvgPool2d(4)
        self.emo_fc_2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Linear(128, self.n_expression + n_reg))

    def forward(self, x, reset_smoothing=False):
        
        if self.init_smoothing:
            self.init_smoothing = False
            self.temporal_state = torch.zeros(x.size(0), self.n_temporal_states, self.n_expression+self.n_reg).cuda()              
        if reset_smoothing:
            self.temporal_state = self.temporal_state.zeros_() 

        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.max_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        hg_features = []

        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            tmp_out = self._modules['l' + str(i)](ll)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

            hg_features.append(ll)

        hg_features_cat = torch.cat(tuple(hg_features), dim=1)

        if self.attention:
            mask = torch.sum(tmp_out, dim=1, keepdim=True)
            hg_features_cat *= mask
            emo_feat = torch.cat((x, hg_features_cat), dim=1)
        else:
            emo_feat = torch.cat([x, hg_features_cat, tmp_out], dim=1)
        
        emo_feat_conv1D = self.conv1x1_input_emo_2(emo_feat)
        final_features = self.emo_net_2(emo_feat_conv1D)
        final_features = self.avg_pool_2(final_features)
        batch_size = final_features.shape[0]
        final_features = final_features.view(batch_size, final_features.shape[1])
        final_features = self.emo_fc_2(final_features)
        
        if self.temporal_smoothing:
            with torch.no_grad():
                self.temporal_state[:,:-1,:] = self.temporal_state[:,1:,:]
                self.temporal_state[:,-1,:] = final_features 
                final_features = torch.sum(self.temporal_weights*self.temporal_state, dim=1)

        return {'heatmap': tmp_out, 'expression': final_features[:,:-2], 'valence': final_features[:,-2], 'arousal':final_features[:,-1]}

  
    def eval(self):
        
        for module in self.children():
            module.eval()


class EmoNetClassifier:
    """情绪识别分类器"""
    
    def __init__(self):
        logger.info("正在加载 EmoNet 表情分类器...")
        self.device = config.DEVICE
        self.model_path = config.EMOTION_MODEL_PATH
        self.input_size = (256, 256)
        self.emotion_labels = config.EMOTION_LIST
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"EmoNet 权重文件未找到: {self.model_path}")

        self.model = EmoNet(n_expression=len(self.emotion_labels))
        self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("EmoNet 表情分类器加载完成。")
        
    def _load_model(self):
        logger.info(f"正在从 {self.model_path} 加载权重...")
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        # 处理多卡模型的前缀
        if 'module.' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict=True)
        else:
            self.model.load_state_dict(state_dict, strict=True)

    def _preprocess_batch(self, face_batch: list) -> torch.Tensor:
        """预处理人脸批次"""
        processed_tensors = []
        for face_img in face_batch:
            img = cv2.resize(face_img, self.input_size, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.stack([img, img, img], axis=-1)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            processed_tensors.append(torch.from_numpy(img))
        
        batch_tensor = torch.stack(processed_tensors).to(self.device)
        return batch_tensor

    def predict_batch(self, face_batch: list) -> list:
        """批量预测情绪"""
        if not face_batch: 
            return []
            
        batch_tensor = self._preprocess_batch(face_batch)
        with torch.no_grad():
            predictions = self.model(batch_tensor)
            expression = predictions['expression']
            valence = predictions['valence']
            arousal = predictions['arousal']
            probs = F.softmax(expression, dim=1).cpu().numpy()
            best_match_indices = np.argmax(probs, axis=1)
            valence = valence.cpu().numpy().flatten()
            arousal = arousal.cpu().numpy().flatten()

        results = []
        for i in range(len(face_batch)):
            results.append({
                'emotion': self.emotion_labels[best_match_indices[i]],
                'valence': float(valence[i]),
                'arousal': float(arousal[i])
            })
        return results
    
    def predict_single(self, face_img: np.ndarray) -> dict:
        """单个人脸预测"""
        results = self.predict_batch([face_img])
        return results[0] if results else None