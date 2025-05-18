import torch
from torch import nn
import torch.nn.functional as F
from .models import New_Audio_Guided_Attention, AVGA, LSTM_A_V, MWTF, EGTA, rak_module
from .models import EncoderLayer, Encoder, DecoderLayer, Decoder
from torch.nn import MultiheadAttention

import pdb
import numpy as np


torch.set_default_dtype(torch.float32)

class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.ReLU = nn.ReLU(inplace=True)
        # add ReLU here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature


class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)

        return output


class WeaklyLocalizationModule(nn.Module):
    def __init__(self, input_dim):
        super(WeaklyLocalizationModule, self).__init__()

        self.hidden_dim = input_dim # need to equal d_model
        self.classifier = nn.Linear(self.hidden_dim, 1) # start and end
        self.event_classifier = nn.Linear(self.hidden_dim, 29)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        fused_content = fused_content.transpose(0, 1)
        max_fused_content, _ = fused_content.max(1)
        # confident scores
        is_event_scores = self.classifier(fused_content)
        # classification scores
        raw_logits = self.event_classifier(max_fused_content)[:, None, :]
        # fused
        fused_logits = is_event_scores.sigmoid() * raw_logits
        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        return is_event_scores.squeeze(), raw_logits.squeeze(), event_scores


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.ReLU = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model, 1) # start and end
        self.event_classifier = nn.Linear(d_model, 28)
       # self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        logits = self.classifier(fused_content) # torch.Size([10, 32, 1024]) -> torch.Size([10, 32, 1])
        # scores = self.softmax(logits)
        class_logits = self.event_classifier(max_fused_content) # torch.Size([32, 1024]) -> torch.Size([32, 28])
        class_scores = class_logits

        return logits, class_scores


class AudioVideoInter(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudioVideoInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)


    def forward(self, video_feat, audio_feat):
        # video_feat, audio_feat: [10, batch, 256]
        global_feat = video_feat * audio_feat
        memory = torch.cat([audio_feat, video_feat], dim=0)
        mid_out = self.video_multihead(global_feat, memory, memory)[0]
        output = self.norm1(global_feat + self.dropout(mid_out))

        return  output
    


class supv_main_model(nn.Module):
    def __init__(self):
        super(supv_main_model, self).__init__()

        self.spatial_channel_att = New_Audio_Guided_Attention().cuda()
        self.video_input_dim = 1024 
        self.video_fc_dim = 1024
        self.d_model = 256
        self.output_dim = 256
        self.num_categories = 28
        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.a_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.agva = AVGA(a_dim=1024, v_dim=1024)
        self.lstm_a_v = LSTM_A_V(a_dim=1024, v_dim=1024, hidden_size=128)
        self.mwtf = MWTF(input_dim=512,hidden_size=128)
        self.egta = EGTA(input_dim=512,hidden_size=512)
        # self.egta = EGTA(input_dim=256,hidden_size=128)
        self.refiner_mwtf = MWTF(input_dim=512,hidden_size=512)
        self.refiner_lstm = nn.LSTM(512, self.output_dim, bidirectional=True, batch_first=True)
        self.refiner_w = nn.Linear(self.output_dim * 2, self.num_categories)
        
        self.ln128 = nn.LayerNorm(128)
        self.ln256 = nn.LayerNorm(256)
        self.ln512 = nn.LayerNorm(512)
        self.ln1024 = nn.LayerNorm(1024)
        self.localize_module = SupvLocalizeModule(d_model=1024)
        
        self.cos = nn.CosineSimilarity(dim = -1)
        self.tanh = nn.Tanh()
        self.batch_norm = nn.BatchNorm1d(num_features=10)
        self.weight = 0.5



    def forward(self, visual_feature, audio_feature, file_names = [], epoch = -1):

        """유사도"""
        # 코사인
        segment_similarity = self.cos(visual_feature, audio_feature)
        segment_similarity = torch.unsqueeze(segment_similarity, -1)
        
        # # audio_feature = audio_feature.transpose(1, 0).contiguous()  # -> torch.Size([10, 32, 1024])
        visual_feature = self.v_fc(visual_feature)                  # -> torch.Size([32, 10, 1024])
        # audio_feature = self.a_fc(audio_feature)                  # -> torch.Size([32, 10, 1024])
        visual_feature = self.dropout(self.ReLU(visual_feature))    # -> torch.Size([32, 10, 1024])
        # audio_feature = self.dropout(self.ReLU(audio_feature))    # -> torch.Size([32, 10, 1024])
        
        '''Audio guided visual attention'''
        visual_feature = self.agva(audio_feature, visual_feature)    # -> torch.Size([32, 10, 512])
        
        '''BI-LSTM'''
        # input : (T, 1024) -> output : (T, 256)1
        lstm_audio, lstm_video = self.lstm_a_v(audio_feature, visual_feature)
        concat_va = torch.cat((lstm_video, lstm_audio), dim=-1)
        
        '''Multi-window Temporal Fusion'''
        feat1 = self.dropout(self.mwtf(concat_va, window=10, file_names=file_names, epoch=epoch))
        feat2 = self.dropout(self.mwtf(concat_va, window=5))
        feat3 = self.dropout(self.mwtf(concat_va,  window=3))
        feat4 = self.dropout(self.mwtf(concat_va,  window=2, file_names=file_names, epoch=epoch))
        mwtf_feat = torch.cat((feat1, feat2, feat3, feat4), dim=-1)
        
        
        '''EGTA'''
        alpha_att = self.egta(mwtf_feat)
        
        """normalization"""
        # 선형변환
        segment_similarity = (segment_similarity + 1) / 2
        # segment_similarity = segment_similarity.sigmoid() # sigmoid
        # segment_similarity = (segment_similarity - segment_similarity.min()) / (segment_similarity.max() - segment_similarity.min()) # min-max
        # segment_similarity = (self.tanh(segment_similarity) + 1) / 2 # tanh
        
        """원래"""
        event_relevant = ((1-self.weight) * alpha_att) + (self.weight * segment_similarity)

        
        '''Refiner'''
        # refine_feat = mwtf_feat * event_relevant
        refine_feat = mwtf_feat * alpha_att

        # Single window fusion over the refined vector O_prime        
        refine_feat = self.refiner_mwtf(refine_feat, window=10) 
        refine_feat = self.ln512(refine_feat)

        max_feat, _ = refine_feat.max(1) # torch.Size([batch, dimension])
        class_logits = F.softmax(self.refiner_w(max_feat), dim=-1)
        
        event_relevant = event_relevant.transpose(1,0).contiguous()
        scores = (event_relevant, class_logits, refine_feat, alpha_att, segment_similarity)

        return scores

        