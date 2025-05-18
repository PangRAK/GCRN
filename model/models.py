import torch
import copy
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
from torch.nn import init

import pdb


class Encoder(Module):
    r"""Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        r"""Pass the input through the endocder layers in turn.

        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output)

        if self.norm:
            output = self.norm(output)

        return output


class Decoder(Module):
    r"""Decoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the DecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory)

        if self.norm:
            output = self.norm(output)

        return output


class EncoderLayer(Module):
    r"""EncoderLayer is mainly made up of self-attention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src):
        r"""Pass the input through the endocder layer.
        """
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DecoderLayer(Module):
    r"""DecoderLayer is mainly made up of the proposed cross-modal relation attention (CMRA).

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer.
        """
        memory = torch.cat([memory, tgt], dim=0)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class New_Audio_Guided_Attention(nn.Module):
    def __init__(self):
        super(New_Audio_Guided_Attention, self).__init__()
        self.hidden_size = 512
        self.relu = nn.ReLU()
        # channel attention
        self.affine_video_1 = nn.Linear(512, 512)
        self.affine_audio_1 = nn.Linear(128, 512)
        self.affine_bottleneck = nn.Linear(512, 256)
        self.affine_v_c_att = nn.Linear(256, 512)
        # spatial attention
        self.affine_video_2 = nn.Linear(512, 256)
        self.affine_audio_2 = nn.Linear(128, 256)
        self.affine_v_s_att = nn.Linear(256, 1)

        # video-guided audio attention
        self.affine_video_guided_1 = nn.Linear(512, 64)
        self.affine_video_guided_2 = nn.Linear(64, 128)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, video, audio):
        '''
        :param visual_feature: [batch, 10, 7, 7, 512]
        :param audio_feature:  [batch, 10, 128]
        :return: [batch, 10, 512]
        '''
        audio = audio.transpose(1, 0)
        batch, t_size, h, w, v_dim = video.size()
        a_dim = audio.size(-1)
        audio_feature = audio.reshape(batch * t_size, a_dim)
        visual_feature = video.reshape(batch, t_size, -1, v_dim)
        raw_visual_feature = visual_feature
        # ============================== Channel Attention ====================================
        audio_query_1 = self.relu(self.affine_audio_1(audio_feature)).unsqueeze(-2)
        video_query_1 = self.relu(self.affine_video_1(visual_feature)).reshape(batch*t_size, h*w, -1)
        audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2)
        audio_video_query = self.relu(self.affine_bottleneck(audio_video_query_raw))
        channel_att_maps = self.affine_v_c_att(audio_video_query).sigmoid().reshape(batch, t_size, -1, v_dim)
        c_att_visual_feat = (raw_visual_feature * (channel_att_maps + 1))

        # ============================== Spatial Attention =====================================
        # channel attended visual feature: [batch * 10, 49, v_dim]
        c_att_visual_feat = c_att_visual_feat.reshape(batch*t_size, -1, v_dim)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat))
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(-2)
        audio_video_query_2 = c_att_visual_query * audio_query_2
        spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)).transpose(2, 1))
        c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat).squeeze().reshape(batch, t_size, v_dim)

        return c_s_att_visual_feat

# RAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAKRAK
class LSTM_A_V(nn.Module):
    """Bi-LSTM is utilized to encode temporal relations in video segments.
    Zhou, Jinxing, et al. "Positive sample propagation along the audio-visual event line." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
    """
    def __init__(self, a_dim=1024, v_dim=1024, hidden_size=128, seg_num=10):
        super(LSTM_A_V, self).__init__()

        self.lstm_audio = nn.LSTM(a_dim, hidden_size, 1, batch_first=True, bidirectional=True, dropout=0.0)
        self.lstm_video = nn.LSTM(v_dim, hidden_size, 1, batch_first=True, bidirectional=True, dropout=0.0)

    def init_hidden(self, a_dea, v_fea):
        bs, seg_num, a_dim = a_dea.shape
        hidden_a = (torch.zeros(2, bs, 128).cuda(), torch.zeros(2, bs, 128).cuda())
        hidden_v = (torch.zeros(2, bs, 128).cuda(), torch.zeros(2, bs, 128).cuda())
        return hidden_a, hidden_v

    def forward(self, a_dea, v_fea):
        # a_dea, v_fea: [bs, 10, 128]
        hidden_a, hidden_v = self.init_hidden(a_dea, v_fea)
        # Bi-LSTM for temporal modeling
        self.lstm_video.flatten_parameters() # .contiguous()
        self.lstm_audio.flatten_parameters()
        lstm_audio, hidden1 = self.lstm_audio(a_dea, hidden_a)
        lstm_video, hidden2 = self.lstm_video(v_fea, hidden_v)

        return lstm_audio, lstm_video
 
class AVGA(nn.Module):
    """Audio-guided visual attention used in AVEL.
    AVEL:Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chen-liang Xu. Audio-visual event localization in unconstrained videos. InECCV, 2018
    """
    def __init__(self, a_dim=1024, v_dim=1024, hidden_size=1024, map_size=1024):
        super(AVGA, self).__init__()
        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(a_dim, hidden_size)
        self.affine_video = nn.Linear(v_dim, hidden_size)
        self.affine_v = nn.Linear(hidden_size, map_size, bias=False)
        self.affine_g = nn.Linear(hidden_size, map_size, bias=False)
        self.affine_h = nn.Linear(map_size, map_size, bias=False)
        self.ln1024 = nn.LayerNorm(1024)

        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)
        init.xavier_uniform(self.affine_audio.weight)
        init.xavier_uniform(self.affine_video.weight)

    def forward(self, audio, video):        
        # audio : torch.Size([10, 32, 1024])
        # video : torch.Size([32, 10, 1024])
        """
        1.
        현재 코드는 channel dimension에 대한 attention만 진행하는데
        Time-level attention도 해야하는지 확인 필요함 -> (32,10,1024)로 해야할듯
        
        2. 
        Time-level attention은 아니더라도 (32,10,1024)로 해야하는거 아닌지?
        """
        
        # audio = audio.transpose(1, 0).contiguous() # audio : torch.Size([32, 10, 1024])
        
        V_DIM = video.size(-1) # 1024
        v_t = video.view(video.size(0) * video.size(1), V_DIM) # torch.Size([320, 1024])
        V = v_t # torch.Size([320, 1024])

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t)) # torch.Size([320, 1, 512])
        a_t = audio.view(-1, audio.size(-1)) # torch.Size([320, 1024])
        a_t = self.relu(self.affine_audio(a_t)) # torch.Size([320, 512])
        content_v = self.affine_v(v_t) + self.affine_g(a_t) # [bs*10, 49, 49] + [bs*10, 49, 1]

        z_t = self.affine_h((F.tanh(content_v))) # torch.Size([320, 1024]) / [bs*10, 49]
        alpha_t = F.softmax(z_t, dim=-1) # torch.Size([320, 1, 1024]) / attention map, [bs*10, 1, 49]
        # c_t = torch.matmul(alpha_t, V) # [bs*10, 1, 512]
        
        c_t = alpha_t * V
        # c_t = V + (alpha_t * V) # residual
        
        video_t = c_t.view(video.size(0), -1, V_DIM) # attended visual features, [bs, 10, 512]
        video_t = self.ln1024(video_t)
        
        return video_t
    
class LSTM_A_V(nn.Module):
    """Bi-LSTM is utilized to encode temporal relations in video segments.
    Zhou, Jinxing, et al. "Positive sample propagation along the audio-visual event line." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
    """
    def __init__(self, a_dim=1024, v_dim=1024, hidden_size=128, seg_num=10):
        super(LSTM_A_V, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_audio = nn.LSTM(a_dim, hidden_size, 1, batch_first=True, bidirectional=True, dropout=0.0)
        self.lstm_video = nn.LSTM(v_dim, hidden_size, 1, batch_first=True, bidirectional=True, dropout=0.0)

    def init_hidden(self, a_dea, v_fea):
        bs, seg_num, a_dim = a_dea.shape
        hidden_a = (torch.zeros(2, bs, self.hidden_size).cuda(), torch.zeros(2, bs, self.hidden_size).cuda())
        hidden_v = (torch.zeros(2, bs, self.hidden_size).cuda(), torch.zeros(2, bs, self.hidden_size).cuda())
        return hidden_a, hidden_v

    def forward(self, a_dea, v_fea):
        # a_dea, v_fea: [bs, 10, 128]
        hidden_a, hidden_v = self.init_hidden(a_dea, v_fea)
        # Bi-LSTM for temporal modeling
        self.lstm_video.flatten_parameters() # .contiguous()
        self.lstm_audio.flatten_parameters() 
        lstm_audio, hidden1 = self.lstm_audio(a_dea, hidden_a)
        lstm_video, hidden2 = self.lstm_video(v_fea, hidden_v)

        return lstm_audio, lstm_video

class MWTF(nn.Module):
    def __init__(self, input_dim, hidden_size = 512):
        super(MWTF, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.w1 = nn.Linear(input_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.w3 = nn.Linear(input_dim, hidden_size)
        self.w4 = nn.Linear(hidden_size, hidden_size)
        self.w5 = nn.Linear(input_dim, hidden_size)
        self.w6 = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size
        
        init.xavier_uniform(self.w1.weight)
        init.xavier_uniform(self.w2.weight)
        init.xavier_uniform(self.w3.weight)
        init.xavier_uniform(self.w4.weight)
        init.xavier_uniform(self.w5.weight)
        init.xavier_uniform(self.w6.weight)

    def forward(self, feature, window, file_names = [], epoch = -1):
        # if (epoch > 50) and (bytes('3ZqnmeJ_ubA', 'utf-8') in file_names):
        #     pdb.set_trace()
        
        res = None
            
        splt_feat = feature.split(window, dim=-2)
        splt_feat_list = list(splt_feat)
        
        if window == 3:
            splt_feat_list[2] = torch.cat((splt_feat_list[2], splt_feat_list[3]), dim=-2)
            splt_feat_list = splt_feat_list[0:3]
        for block in splt_feat_list:
            norm_feat = self.norm(block)
            
            Q_feat = self.w2(torch.tanh(self.w1(norm_feat)))
            K_feat = self.w4(torch.tanh(self.w3(norm_feat)))
            V_feat = self.w6(F.relu(self.w5(norm_feat)))
            
            sqrt_feat = torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
            sqrt_time = torch.sqrt(torch.tensor(window, dtype=torch.float32))
            beta_time = torch.bmm(Q_feat, K_feat.transpose(1, 2)) / sqrt_feat
            beta_feat = torch.bmm(Q_feat.transpose(1, 2), K_feat) / sqrt_time
            
            # Apply softmax to generate attention maps
            """Transpose 필요한지, softmax dim 맞는지, 검토 필요!!!!!!!"""
            A_time = F.softmax(beta_time, dim=-1)  # Temporal attention map
            A_feat = F.softmax(beta_feat, dim=-2)  # Feature attention map

            # Apply the attention maps to V_i_k
            feat = torch.bmm(V_feat, A_feat) 
            feat = torch.bmm(A_time, feat)
            if res == None:
                res = feat
            else:
                res = torch.cat((res, feat), dim=-2) # Time-level concate

        return res
    
class EGTA(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(EGTA, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_size, bidirectional=True, batch_first=True)
        self.W = nn.Linear(hidden_size * 2, 1)
        self.input_dim = input_dim
        self.hidden_size = hidden_size


    def forward(self, O):
        O_norm = self.norm(O)
        gamma, _ = self.lstm(O_norm)
        
        # Generate EGTA mask alpha_att
        alpha_att = torch.sigmoid(self.W(gamma))

        return alpha_att
 
class rak_module(nn.Module):
    def __init__(self, input_dim, hidden_size = 512):
        super(rak_module, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.flt_norm = nn.LayerNorm(input_dim * 10)
        self.wf_1 = nn.Linear(input_dim, hidden_size)
        self.wf_2 = nn.Linear(hidden_size, hidden_size)
        self.wf_3 = nn.Linear(input_dim, hidden_size)
        self.wf_4 = nn.Linear(hidden_size, hidden_size)
        self.wf_5 = nn.Linear(input_dim, hidden_size)
        self.wf_6 = nn.Linear(hidden_size, hidden_size)
        self.wt_1 = nn.Linear(input_dim, hidden_size)
        self.wt_2 = nn.Linear(hidden_size, hidden_size)
        self.wt_3 = nn.Linear(input_dim, hidden_size)
        self.wt_4 = nn.Linear(hidden_size, hidden_size)
        
        self.att_fusion_1 = nn.Linear(input_dim * 10, hidden_size)
        self.att_fusion_2 = nn.Linear(hidden_size, 2)
        self.att_fusion_3 = nn.Linear(input_dim * 10, hidden_size)
        self.att_fusion_4 = nn.Linear(hidden_size, 2)
        
        self.fusion_ln = nn.Linear(2, 1)
        self.hidden_size = hidden_size
        
        init.xavier_uniform(self.wf_1.weight)
        init.xavier_uniform(self.wf_2.weight)
        init.xavier_uniform(self.wf_3.weight)
        init.xavier_uniform(self.wf_4.weight)
        init.xavier_uniform(self.wt_1.weight)
        init.xavier_uniform(self.wt_2.weight)
        init.xavier_uniform(self.wt_3.weight)
        init.xavier_uniform(self.wt_4.weight)
        # init.xavier_uniform(self.wt_5.weight)
        # init.xavier_uniform(self.wt_6.weight)

    def forward(self, feature, window):
        # flatten_feat = torch.flatten(feature, start_dim=1)
        # flatten_feat = self.flt_norm(flatten_feat)
        # fusion_Q = self.att_fusion_2(torch.tanh(self.att_fusion_1(torch.flatten(feature, start_dim=1))))
        # fusion_K = self.att_fusion_4(torch.tanh(self.att_fusion_3(torch.flatten(feature, start_dim=1))))
        # sqrt_fusion = torch.sqrt(torch.tensor(2, dtype=torch.float32))
        # time_energy = torch.bmm(fusion_Q, fusion_K.transpose(1, 2)) / sqrt_time
        res = None
            
        splt_feat = feature.split(window, dim=-2)
        splt_feat_list = list(splt_feat)
        
        if window == 3:
            splt_feat_list[2] = torch.cat((splt_feat_list[2], splt_feat_list[3]), dim=-2)
            splt_feat_list = splt_feat_list[0:3]
        for block in splt_feat_list:
            norm_feat = self.norm(block)
            
            V_feat = self.wf_6(F.relu(self.wf_5(norm_feat)))      
            feat_Q_feat = self.wf_2(torch.tanh(self.wf_1(norm_feat)))
            feat_K_feat = self.wf_4(torch.tanh(self.wf_3(norm_feat)))      
            time_Q_feat = self.wt_2(torch.tanh(self.wt_1(norm_feat)))
            time_K_feat = self.wt_4(torch.tanh(self.wt_3(norm_feat)))
            
            sqrt_feat = torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
            sqrt_time = torch.sqrt(torch.tensor(window, dtype=torch.float32))
            
            feat_energy = torch.bmm(feat_Q_feat.transpose(1, 2), feat_K_feat) / sqrt_time
            time_energy = torch.bmm(time_Q_feat, time_K_feat.transpose(1, 2)) / sqrt_feat
            
            # Apply softmax to generate attention maps
            """Transpose 필요한지, softmax dim 맞는지, 검토 필요!!!!!!!"""
            time_attention = F.softmax(time_energy, dim=-1)  # Temporal attention map
            feat_attention = F.softmax(feat_energy, dim=-2)  # Feature attention map
            
            # Apply the attention maps to V_i_k
            feat = torch.bmm(V_feat, feat_attention) 
            feat = torch.bmm(time_attention, feat)
            
            if res == None:
                res = feat
            else:
                res = torch.cat((res, feat), dim=-2) # Time-level concate
                
        return res