import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from visual_net import resnet18
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from tqdm.auto import tqdm
from torchvision.transforms import Resize
from torch.nn.functional import interpolate
from PIL import Image
import os

def batch_organize(out_match_posi, out_match_nega):
    # audio B 512
    # posi B 512
    # nega B 512

    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0
    
    return out_match, batch_labels

# Question
class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature

class AVQA_Memory(nn.Module):
    def __init__(self, memory_size, faeture_dim):
        super(AVQA_Memory, self).__init__()
        self.memory_size = memory_size
        self.feature_dim = faeture_dim
        
        """ self.question_encoder = nn.Linear(self.feature_dim, self.feature_dim*2)
        self.question_embedding = nn.Linear(self.feature_dim*2, self.feature_dim)
        
        self.audio_encoder = nn.Linear(self.feature_dim, self.feature_dim*2)
        self.audio_embedding = nn.Linear(self.feature_dim*2, self.feature_dim)
        
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(faeture_dim, faeture_dim*2, kernel_size=3),
            nn.ReLU())
        self.visual_embedding = nn.Sequential(
            nn.ConvTranspose2d(faeture_dim*2, faeture_dim, kernel_size=3),
            nn.ReLU()) """
        
        self.question_memory = torch.empty((self.memory_size, self.feature_dim))
        self.question_memory = nn.init.normal_(self.question_memory, mean=0, std=1)
        self.question_memory = nn.Parameter(self.question_memory, requires_grad=True)
        
        self.audio_memory = torch.empty((self.memory_size, self.feature_dim))
        self.audio_memory = nn.init.normal_(self.audio_memory, mean=0, std=1)
        self.audio_memory = nn.Parameter(self.audio_memory, requires_grad=True)
        
        self.visual_memory = torch.empty((self.memory_size, self.feature_dim*14*14))
        self.visual_memory = nn.init.normal_(self.visual_memory, mean=0, std=1)
        self.visual_memory = nn.Parameter(self.visual_memory, requires_grad=True)
    
    def forward(self, audio, visual, question):
        
        question = question.repeat(10, 1, 1) # [10, B, C]
        Q_B, Q_T, Q_C = question.shape  # batch, time step, channel of the Question
        question = question.reshape(Q_B*Q_T, Q_C) # [640, 512]
        question_norm = nn.functional.normalize(question, dim=1)
        question_memory_norm = nn.functional.normalize(self.question_memory, dim=1)
        Mem_Q = torch.mm(question_norm, question_memory_norm.transpose(0, 1)) # [640, M_SIZE]
        addressing_Q = nn.functional.softmax(Mem_Q, dim=-1) # [640, M_SIZE]     M_SIZE means Memory Size
        
        if self.training:
            A_B, A_T, A_C = audio.shape
            audio = audio.reshape(A_B*A_T, A_C) # [640, 512]
            audio_norm = nn.functional.normalize(audio, dim=1)
            audio_memory_norm = nn.functional.normalize(self.audio_memory, dim=1)
            Mem_A = torch.mm(audio_norm, audio_memory_norm.transpose(0, 1)) # [640, M_SIZE]
            addressing_A = nn.functional.softmax(Mem_A, dim=-1) # [640, M_SIZE]
            
            V_B, V_T, V_C, V_H, V_W = visual.shape
            visual = visual.reshape(V_B*V_T, V_C*V_H*V_W) # [640, 512*14*14]
            visual_norm = nn.functional.normalize(visual, dim=1)
            visual_memory_norm = nn.functional.normalize(self.visual_memory, dim=1)
            Mem_V = torch.mm(visual_norm, visual_memory_norm.transpose(0, 1)) # [640, M_SIZE]
            addressing_V = nn.functional.softmax(Mem_V, dim=-1) # [640, M_SIZE]
            
            addressing_QV = torch.mul(addressing_Q, addressing_V) # [640, M_SIZE]
            addressing_QV = nn.functional.softmax(addressing_QV, dim=-1) # [640, M_SIZE]
            pseudo_audio = torch.mm(addressing_QV, self.audio_memory) # [640, 512]
            pseudo_audio = torch.reshape(pseudo_audio, (A_B, A_T, A_C)) # [B, T, C]
            
            addressing_QA = torch.mul(addressing_Q, addressing_A) # [640, M_SIZE]
            addressing_QA = nn.functional.softmax(addressing_QA, dim=-1) # [640, M_SIZE]
            pseudo_visual = torch.mm(addressing_QA, self.visual_memory) # [640, 512*14*14]
            pseudo_visual = torch.reshape(pseudo_visual, (V_B, V_T, V_C, V_H, V_W)) # [B, T, C, H, W]
            
            # kyuri memory_loss
            # loss 사이즈 안 맞음

            visual = visual.reshape(V_B,V_T, V_C,V_H,V_W) # [640, 512*14*14]
            audio = audio.reshape(A_B, A_T, A_C) # [640, 512]
            """ print(visual.shape) #torch.Size([80, 100352])
            print(pseudo_visual.shape)  #torch.Size([8, 10, 512, 14, 14])
            print(audio.shape)  #torch.Size([80, 512])
            print(pseudo_audio.shape)   #torch.Size([8, 10, 512]) """
            loss_visual = F.mse_loss(pseudo_visual, visual)
            loss_audio = F.mse_loss(pseudo_audio, audio)
            memory_loss = loss_visual + loss_audio
            
            return pseudo_audio, pseudo_visual, memory_loss
        
        else:
            if audio is None:
                V_B, V_T, V_C, V_H, V_W = visual.shape
                visual = visual.reshape(V_B*V_T, V_C*V_H*V_W) # [640, 512*14*14]
                visual_norm = nn.functional.normalize(visual, dim=1)
                visual_memory_norm = nn.functional.normalize(self.visual_memory, dim=1)
                Mem_V = torch.mm(visual_norm, visual_memory_norm.transpose(0, 1)) # [640, M_SIZE]
                addressing_V = nn.functional.softmax(Mem_V, dim=-1) # [640, M_SIZE]
                
                addressing_QV = torch.mul(addressing_Q, addressing_V) # [640, M_SIZE]
                addressing_QV = nn.functional.softmax(addressing_QV, dim=-1) # [640, M_SIZE]
                pseudo_audio = torch.mm(addressing_QV, self.audio_memory) # [640, 512]
                pseudo_audio = torch.reshape(pseudo_audio, (V_B, V_T, V_C)) # [B, T, C]
                
                return pseudo_audio
            
            elif visual is None:
                A_B, A_T, A_C = audio.shape
                audio = audio.reshape(A_B*A_T, A_C) # [640, 512]
                audio_norm = nn.functional.normalize(audio, dim=1)
                audio_memory_norm = nn.functional.normalize(self.audio_memory, dim=1)
                Mem_A = torch.mm(audio_norm, audio_memory_norm.transpose(0, 1)) # [640, M_SIZE]
                addressing_A = nn.functional.softmax(Mem_A, dim=-1) # [640, M_SIZE]
                
                addressing_QA = torch.mul(addressing_Q, addressing_A) # [640, M_SIZE]
                addressing_QA = nn.functional.softmax(addressing_QA, dim=-1) # [640, M_SIZE]
                pseudo_visual = torch.mm(addressing_QA, self.visual_memory) # [640, 512*14*14]
                H = int(math.sqrt(pseudo_visual.shape[1]/self.feature_dim))
                W = H
                pseudo_visual = torch.reshape(pseudo_visual, (A_B, A_T, A_C, H, W)) # [B, T, C, H, W]
                
                return pseudo_visual
            
            else:
                raise ValueError('Set missing_sutuation')
        
    
class AVQA_Fusion_Net(nn.Module):

    def __init__(self, args):
        super(AVQA_Fusion_Net, self).__init__()

        # for features
        self.fc_a1 =  nn.Linear(128, 512)
        self.fc_a2=nn.Linear(512,512)

        self.fc_a1_pure =  nn.Linear(128, 512)
        self.fc_a2_pure=nn.Linear(512,512)
        self.visual_net = resnet18(pretrained=True)

        self.fc_v = nn.Linear(2048, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)
        self.fc = nn.Linear(1024, 512)
        self.fc_aq = nn.Linear(512, 512)
        self.fc_vq = nn.Linear(512, 512)

        self.linear11 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.linear12 = nn.Linear(512, 512)

        self.linear21 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.1)
        self.linear22 = nn.Linear(512, 512)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(512)

        self.attn_a = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.attn_v = nn.MultiheadAttention(512, 4, dropout=0.1)

        # question
        self.question_encoder = QstEncoder(93, 512, 512, 1, 512)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc_ans = nn.Linear(512, 42)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_gl=nn.Linear(1024,512)

        # combine
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 2)
        self.relu4 = nn.ReLU()
        self.kr = nn.Conv2d(1024,512,1)
        self.kr2 = nn.Conv2d(512,1024,1)

        self.conv_sm = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        self.h = None
        self.w = None 
        # self.diffusion_()
        self.printed = False
        self.num_timesteps = args.time_step  # number of steps = 1000 원래 1000임.
        self.diffusion_init()
        
        self.memory_network = AVQA_Memory(args.memory_size, 512)
        self.default_missing_situation = args.missing_situation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        self.missing_situation = self.default_missing_situation
    
    def diffusion_init(self):
        self.unet = Unet(
            dim=64,
            dim_mults=(1, 2),
            channels=512
        )
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            image_size=14,
            timesteps=self.num_timesteps,
        )

    # kyuri 
    def diffusion_process(self, x): #kyuri
        diff_loss = self.diffusion(clean_feat=x)
        return diff_loss
        
    def set_missing_situation(self, situation=None):
        if situation is None:
            self.missing_situation = self.default_missing_situation
        else:
            self.missing_situation = situation

    def origin_avqa(self, origin_audio, visual_posi, visual_nega, qst_feature, xq):
        audio_feat_pure = origin_audio
        B, T, C = origin_audio.size()             # [B, T, C]
        audio_feat = origin_audio.view(B*T, C)    # [B*T, C]
        
        ## visual posi [2*B*T, C, H, W]
        B, T, C, H, W = visual_posi.size()
        temp_visual = visual_posi.contiguous().view(B*T, C, H, W)            # [B*T, C, H, W]
        v_feat = self.avgpool(temp_visual)                      # [B*T, C, 1, 1]
        visual_feat_before_grounding_posi = v_feat.squeeze()    # [B*T, C]

        (B, C, H, W) = temp_visual.size()
        v_feat = temp_visual.view(B, C, H * W)                      # [B*T, C, HxW]
        v_feat = v_feat.permute(0, 2, 1)                            # [B*T, HxW, C]
        visual_feat_posi = nn.functional.normalize(v_feat, dim=2)   # [B*T, HxW, C]

        ## audio-visual grounding posi
        audio_feat_aa = audio_feat.unsqueeze(-1)                        # [B*T, C, 1]
        audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)   # [B*T, C, 1]
        x2_va = torch.matmul(visual_feat_posi, audio_feat_aa).squeeze() # [B*T, HxW]

        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)                       # [B*T, 1, HxW]
        visual_feat_grd = torch.matmul(x2_p, visual_feat_posi)

        visual_feat_grd_after_grounding_posi = visual_feat_grd.squeeze()    # [B*T, C]   

        visual_gl = torch.cat((visual_feat_before_grounding_posi, visual_feat_grd_after_grounding_posi),dim=-1)
        visual_feat_grd = self.tanh(visual_gl)
        visual_feat_grd_posi = self.fc_gl(visual_feat_grd)              # [B*T, C]

        feat = torch.cat((audio_feat, visual_feat_grd_posi), dim=-1)    # [B*T, C*2], [B*T, 1024]

        feat = F.relu(self.fc1(feat))       # (1024, 512)
        feat = F.relu(self.fc2(feat))       # (512, 256)
        feat = F.relu(self.fc3(feat))       # (256, 128)
        out_match_posi = self.fc4(feat)     # (128, 2)

        ###############################################################################################
        # visual nega
        B, T, C, H, W = visual_nega.size()
        temp_visual = visual_nega.view(B*T, C, H, W)
        v_feat = self.avgpool(temp_visual)
        visual_feat_before_grounding_nega = v_feat.squeeze() # [B*T, C]

        (B, C, H, W) = temp_visual.size()
        v_feat = temp_visual.view(B, C, H * W)  # [B*T, C, HxW]
        v_feat = v_feat.permute(0, 2, 1)        # [B, HxW, C]
        visual_feat_nega = nn.functional.normalize(v_feat, dim=2)

        ##### av grounding nega
        x2_va = torch.matmul(visual_feat_nega, audio_feat_aa).squeeze()
        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)                       # [B*T, 1, HxW]
        visual_feat_grd = torch.matmul(x2_p, visual_feat_nega)
        visual_feat_grd_after_grounding_nega = visual_feat_grd.squeeze()    # [B*T, C]   

        visual_gl=torch.cat((visual_feat_before_grounding_nega,visual_feat_grd_after_grounding_nega),dim=-1)
        visual_feat_grd=self.tanh(visual_gl)
        visual_feat_grd_nega=self.fc_gl(visual_feat_grd)    # [B*T, C]

        # combine a and v
        feat = torch.cat((audio_feat, visual_feat_grd_nega), dim=-1)   # [B*T, C*2], [B*T, 1024]

        feat = F.relu(self.fc1(feat))       # (1024, 512)
        feat = F.relu(self.fc2(feat))       # (512, 256)
        feat = F.relu(self.fc3(feat))       # (256, 128)
        out_match_nega = self.fc4(feat)     # (128, 2)

        ###############################################################################################

        # out_match=None
        # match_label=None

        B = xq.shape[1] # xq: [1, B, T]
        visual_feat_grd_be = visual_feat_grd_posi.view(B, -1, 512)   # [B, T, 512]
        visual_feat_grd=visual_feat_grd_be.permute(1,0,2)            # [T, B, 512]
        
        ## attention, question as query on visual_feat_grd
        visual_feat_att = self.attn_v(xq, visual_feat_grd, visual_feat_grd, attn_mask=None, key_padding_mask=None)[0].squeeze(0) # [1, B, 512]  --> [B, 512] ## check
        src = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))
        visual_feat_att = visual_feat_att + self.dropout2(src)
        visual_feat_att = self.norm1(visual_feat_att)
    
        # attention, question as query on audio
        audio_feat_be=audio_feat_pure.view(B, -1, 512)
        audio_feat = audio_feat_be.permute(1, 0, 2)
        audio_feat_att = self.attn_a(xq, audio_feat, audio_feat, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
        src = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))
        audio_feat_att = audio_feat_att + self.dropout4(src)
        audio_feat_att = self.norm2(audio_feat_att)
        
        feat = torch.cat((audio_feat_att+audio_feat_be.mean(dim=-2).squeeze(), visual_feat_att+visual_feat_grd_be.mean(dim=-2).squeeze()), dim=-1)
        feat = self.tanh(feat)
        feat = self.fc_fusion(feat)

        ## fusion with question
        combined_feature = torch.mul(feat, qst_feature) 
        
        # combined_feature = feat
        combined_feature = self.tanh(combined_feature)
        out_qa = self.fc_ans(combined_feature)              # [batch_size, ans_vocab_size]
        
        #return out_qa, out_match_posi,out_match_nega
        if self.training:
            return out_qa, out_match_posi,out_match_nega
        else:
            return out_qa, out_match_posi,out_match_nega
        

    
    def forward(self, audio, visual_posi, visual_nega, question):
        '''
            input question shape:    [B, T]
            input audio shape:       [B, T, C]
            input visual_posi shape: [B, T, C, H, W]
            input visual_nega shape: [B, T, C, H, W]
        '''
        # audio : [64, 10, 128]
        # visual_posi : [64, 10, 512, 14, 14]
        
        ## question features
        qst_feature = self.question_encoder(question)
        xq = qst_feature.unsqueeze(0)

        # make feature named 'temp_visual_noise' that has same shape with visual_posi
        """ B, T, C, H, W = visual_posi.size()
        temp_visual_noise = torch.randn(B, T, C, H, W).to(visual_posi.device)   # 그냥 visual_posi 모양 noise 생성
        temp_visual_noise = temp_visual_noise.view(B*T, C, H, W)
        temp_visual_noise = interpolate(temp_visual_noise, size=(14, 14)) """
        B, T, C, H, W = visual_posi.size()

       
        if self.training:
            origin_audio = F.relu(self.fc_a1(audio))
            origin_audio = self.fc_a2(origin_audio) 

            # pseudo feature
            pseudo_audio, pseudo_visual_posi, memory_loss = self.memory_network(origin_audio, visual_posi, xq)
            
            # diff feature
            diff_for_visual = visual_posi.view(B*T, C, H, W)  #(B, T, C, H, W)            

            diff_for_pseudo_visual = pseudo_visual_posi.view(B*T, C, H, W)  #(B, T, C, H, W)

            diff_for_audio = origin_audio.view(B*T, C)
            diff_for_audio = diff_for_audio.unsqueeze(-1)
            diff_for_audio = diff_for_audio.unsqueeze(-1)
            diff_for_audio = diff_for_audio.repeat(1,1,H,W)
            diff_for_audio = diff_for_audio.view(B*T, C, H, W)  #(B, T, C, H, W) ([80, 512, 14, 14])

            diff_for_pseudo_audio = pseudo_audio.view(B*T, C)
            diff_for_pseudo_audio = diff_for_pseudo_audio.unsqueeze(-1)
            diff_for_pseudo_audio = diff_for_pseudo_audio.unsqueeze(-1)
            diff_for_pseudo_audio = diff_for_pseudo_audio.repeat(1,1,H,W)
            diff_for_pseudo_audio = diff_for_pseudo_audio.view(B*T, C, H, W)  #(B, T, C, H, W) ([80, 512, 14, 14])

            # concat feature
            concat_pseudo = torch.cat((diff_for_pseudo_audio, diff_for_pseudo_visual), dim=1) #(B*T, C, H, W) ([80, 1024, 14, 14])
            concat_pseudo = self.kr(concat_pseudo) #(B, T, C, H, W) ([80, 512, 14, 14])

            concat_real = torch.cat((diff_for_audio, diff_for_visual), dim=1) #(B*T, C, H, W) ([80, 1024, 14, 14])
            concat_real = self.kr(concat_real) #(B, T, C, H, W) ([80, 512, 14, 14])

            # real audio, pseudo visual
            concat_vp = torch.cat((diff_for_audio, diff_for_pseudo_visual), dim=1) #(B*T, C, H, W) ([80, 1024, 14, 14])
            concat_vp = self.kr(concat_vp) #(B, T, C, H, W) ([80, 512, 14, 14])
            
            # pseudo audio, real visual
            concat_ap = torch.cat((diff_for_pseudo_audio, diff_for_visual), dim=1) #(B*T, C, H, W) ([80, 1024, 14, 14])
            concat_ap = self.kr(concat_ap) #(B, T, C, H, W) ([80, 512, 14, 14])

            # diffusion training   바꿈
            
            # diff_loss = 0.
            # step = 0.
            #while step < self.num_timesteps:
            diff_loss = self.diffusion_process(concat_real)    #kyuri concat_real -> Noise -> clean feature
                #step += 1

            #temp_feat = self.diffusion.diff_sample(concat_pseudo)    #[B*T, C, H, W] [80, 512, 14, 14]
            temp_feat_vp = self.diffusion.diff_sample(concat_vp)    #[B*T, C, H, W] [80, 512, 14, 14]
            temp_feat_ap = self.diffusion.diff_sample(concat_ap)    #[B*T, C, H, W] [80, 512, 14, 14]

            temp_feat_vp = self.kr2(temp_feat_vp) #[B*T, C, H, W] [80, 1024, 14, 14]
            temp_feat_ap = self.kr2(temp_feat_ap) #[B*T, C, H, W] [80, 1024, 14, 14]
            
            temp_audio_vp, temp_visual_vp = torch.chunk(temp_feat_vp, chunks=2, dim=1)
            temp_audio_ap, temp_visual_ap = torch.chunk(temp_feat_ap, chunks=2, dim=1)

            ### network 두기
            temp_audio_vp = temp_audio_vp.view(B,T, C, H, W)  
            temp_visual_vp = temp_visual_vp.view(B,T, C, H, W)
            temp_audio_vp = temp_audio_vp.select(3,0).select(3,0)  # [B, T, C]

            temp_audio_ap = temp_audio_ap.view(B,T, C, H, W)
            temp_visual_ap = temp_visual_ap.view(B,T, C, H, W)
            temp_audio_ap = temp_audio_ap.select(3,0).select(3,0)  # [B, T, C]


            temp_audio_ap_loss = F.mse_loss(temp_audio_ap, origin_audio)
            temp_visual_ap_loss = F.mse_loss(temp_visual_ap, visual_posi)
            temp_loss_ap = temp_audio_ap_loss + temp_visual_ap_loss

            temp_audio_vp_loss = F.mse_loss(temp_audio_vp, origin_audio)
            temp_visual_vp_loss = F.mse_loss(temp_visual_vp, visual_posi)
            temp_loss_vp = temp_audio_vp_loss + temp_visual_vp_loss

            temp_loss = temp_loss_ap + temp_loss_vp

            avqa_output = {}
            #out_qa, out_match_posi, out_match_nega  = self.origin_avqa(temp_audio, temp_visual, visual_nega, qst_feature, xq)
            avqa_output["out_qa"], avqa_output["out_match_posi"], avqa_output["out_match_nega"]  = self.origin_avqa(origin_audio, visual_posi, visual_nega, qst_feature, xq)
            avqa_output["out_qa_ap"], avqa_output["visual_match_posi_ap"], avqa_output["visual_match_nega_ap"]  = self.origin_avqa(temp_audio_ap, visual_posi, visual_nega, qst_feature, xq)
            avqa_output["out_qa_vp"], avqa_output["visual_match_posi_vp"], avqa_output["visual_match_nega_vp"]  = self.origin_avqa(origin_audio, temp_visual_vp, visual_nega, qst_feature, xq)
            #audio_match_posi,audio_match_nega = visual_match_posi,visual_match_nega
            
            #return out_qa, audio_match_posi, audio_match_nega, visual_match_posi, visual_match_nega, diff_loss, memory_loss
            return avqa_output, diff_loss, memory_loss, temp_loss

        else: # evaluation, test
            if self.missing_situation == 'audio':
                noise_scale = 0.3
                
                origin_audio = F.relu(self.fc_a1(audio))
                origin_audio = self.fc_a2(origin_audio) 
                
                noise_a = noise_scale * torch.randn_like(origin_audio)
                noisy_audio = origin_audio + noise_a
                pseudo_audio = noisy_audio
                
                # pseudo_audio = torch.randn(B, T, C).to(audio.device)

                diff_for_visual = visual_posi.view(B*T, C, H, W)  #(B, T, C, H, W)
                
                # 추가 부분
                diff_for_pseudo_audio = pseudo_audio.view(B*T, C)
                diff_for_pseudo_audio = diff_for_pseudo_audio.unsqueeze(-1)
                diff_for_pseudo_audio = diff_for_pseudo_audio.unsqueeze(-1)
                diff_for_pseudo_audio = diff_for_pseudo_audio.repeat(1,1,H,W)
                diff_for_pseudo_audio = diff_for_pseudo_audio.view(B*T, C, H, W)  #(B, T, C, H, W) ([80, 512, 14, 14])
                ###############################
                concat_ap = torch.cat((diff_for_pseudo_audio, diff_for_visual), dim=1) #(B*T, C, H, W) ([80, 1024, 14, 14])
                concat_ap = self.kr(concat_ap) #(B, T, C, H, W) ([80, 512, 14, 14])

                temp_feat = self.diffusion.diff_sample(concat_ap)    #[B*T, C, H, W] [80, 512, 14, 14]
                temp_feat = self.kr2(temp_feat) #[B*T, C, H, W] [80, 1024, 14, 14]
                
                temp_audio, temp_visual = torch.chunk(temp_feat, chunks=2, dim=1)
                temp_audio = temp_audio.view(B,T, C, H, W)
                temp_visual = temp_visual.view(B,T, C, H, W)
                temp_audio = temp_audio.select(3,0).select(3,0)

                out_qa, out_match_posi,out_match_nega = self.origin_avqa(temp_audio, visual_posi, visual_nega, qst_feature, xq)

                
            elif self.missing_situation == 'visual':  # visual missing
                origin_audio = F.relu(self.fc_a1(audio))
                origin_audio = self.fc_a2(origin_audio) 

                noise_scale = 0.7
                noise_v = noise_scale * torch.randn_like(visual_posi)
                noisy_visual = visual_posi + noise_v
                pseudo_visual_posi = noisy_visual
                # pseudo_visual_posi = torch.randn(B, T, C, H, W).to(visual_posi.device)

                
                diff_for_audio = origin_audio.view(B*T, C)
                diff_for_audio = diff_for_audio.unsqueeze(-1)
                diff_for_audio = diff_for_audio.unsqueeze(-1)
                diff_for_audio = diff_for_audio.repeat(1,1,H,W)
                diff_for_audio = diff_for_audio.view(B*T, C, H, W)  #(B, T, C, H, W) ([80, 512, 14, 14])

                # diffusion training   
                diff_for_visual = pseudo_visual_posi.view(B*T, C, H, W)  #(B, T, C, H, W)
                concat_vp = torch.cat((diff_for_audio, diff_for_visual), dim=1) #(B*T, C, H, W) ([80, 1024, 14, 14])
                concat_vp = self.kr(concat_vp) #(B, T, C, H, W) ([80, 512, 14, 14])
                
                temp_feat = self.diffusion.diff_sample(concat_vp)    #[B*T, C, H, W] [80, 512, 14, 14]
                temp_feat = self.kr2(temp_feat) #[B*T, C, H, W] [80, 1024, 14, 14]

                temp_audio, temp_visual = torch.chunk(temp_feat, chunks=2, dim=1)

                temp_audio = temp_audio.view(B,T, C, H, W)
                temp_visual = temp_visual.view(B,T, C, H, W)
                temp_audio = temp_audio.select(3,0).select(3,0)
                
                out_qa, out_match_posi,out_match_nega = self.origin_avqa(origin_audio, temp_visual, visual_nega, qst_feature, xq) 

            elif self.missing_situation == 'both':
                noise_scale_v = 0.5
                noise_scale_a = 0.5

                noise_v = noise_scale_v * torch.randn_like(visual_posi)
                noisy_visual = visual_posi + noise_v
                visual_posi = noisy_visual

                origin_audio = F.relu(self.fc_a1(audio))
                origin_audio = self.fc_a2(origin_audio) 

                noise_a = noise_scale_a * torch.randn_like(origin_audio)
                noisy_audio = origin_audio + noise_a
                origin_audio = noisy_audio

                diff_for_audio = origin_audio.view(B*T, C)
                diff_for_audio = diff_for_audio.unsqueeze(-1)
                diff_for_audio = diff_for_audio.unsqueeze(-1)
                diff_for_audio = diff_for_audio.repeat(1,1,H,W)
                diff_for_audio = diff_for_audio.view(B*T, C, H, W)
                diff_for_visual = visual_posi.view(B*T, C, H, W)  #(B, T, C, H, W)
                concat_real = torch.cat((diff_for_audio, diff_for_visual), dim=1)
                concat_real = self.kr(concat_real)
                temp_feat = self.diffusion.diff_sample(concat_real)
                temp_feat = self.kr2(temp_feat)
                temp_audio, temp_visual = torch.chunk(temp_feat, chunks=2, dim=1)
                temp_audio = temp_audio.view(B,T, C, H, W)
                temp_visual = temp_visual.view(B,T, C, H, W)
                temp_audio = temp_audio.select(3,0).select(3,0)

                out_qa, out_match_posi,out_match_nega = self.origin_avqa(temp_audio, temp_visual, visual_nega, qst_feature, xq)

            else:
                raise ValueError('Set missing_situation')
        
        
        
            return out_qa, out_match_posi,out_match_nega


        
            
        """ else: # evaluation, test
            if self.missing_situation == 'audio':
                pseudo_audio = self.memory_network(None, visual_posi, xq)
                diff_for_visual = visual_posi.view(B*T, C, H, W)  #(B, T, C, H, W)
                
                # 추가 부분
                diff_for_pseudo_audio = pseudo_audio.view(B*T, C)
                diff_for_pseudo_audio = diff_for_pseudo_audio.unsqueeze(-1)
                diff_for_pseudo_audio = diff_for_pseudo_audio.unsqueeze(-1)
                diff_for_pseudo_audio = diff_for_pseudo_audio.repeat(1,1,H,W)
                diff_for_pseudo_audio = diff_for_pseudo_audio.view(B*T, C, H, W)  #(B, T, C, H, W) ([80, 512, 14, 14])
                ###############################
                concat_ap = torch.cat((diff_for_pseudo_audio, diff_for_visual), dim=1) #(B*T, C, H, W) ([80, 1024, 14, 14])
                concat_ap = self.kr(concat_ap) #(B, T, C, H, W) ([80, 512, 14, 14])

                temp_feat = self.diffusion.diff_sample(concat_ap)    #[B*T, C, H, W] [80, 512, 14, 14]
                temp_feat = self.kr2(temp_feat) #[B*T, C, H, W] [80, 1024, 14, 14]
                
                temp_audio, temp_visual = torch.chunk(temp_feat, chunks=2, dim=1)
                temp_audio = temp_audio.view(B,T, C, H, W)
                temp_visual = temp_visual.view(B,T, C, H, W)
                temp_audio = temp_audio.select(3,0).select(3,0)

                out_qa, out_match_posi,out_match_nega = self.origin_avqa(temp_audio, visual_posi, visual_nega, qst_feature, xq)

                
            elif self.missing_situation == 'visual':  # visual missing
                origin_audio = F.relu(self.fc_a1(audio))
                origin_audio = self.fc_a2(origin_audio) 
                pseudo_visual_posi = self.memory_network(origin_audio, None, xq)
                
                diff_for_audio = origin_audio.view(B*T, C)
                diff_for_audio = diff_for_audio.unsqueeze(-1)
                diff_for_audio = diff_for_audio.unsqueeze(-1)
                diff_for_audio = diff_for_audio.repeat(1,1,H,W)
                diff_for_audio = diff_for_audio.view(B*T, C, H, W)  #(B, T, C, H, W) ([80, 512, 14, 14])

                # diffusion training   
                diff_for_visual = pseudo_visual_posi.view(B*T, C, H, W)  #(B, T, C, H, W)
                concat_vp = torch.cat((diff_for_audio, diff_for_visual), dim=1) #(B*T, C, H, W) ([80, 1024, 14, 14])
                concat_vp = self.kr(concat_vp) #(B, T, C, H, W) ([80, 512, 14, 14])
                
                temp_feat = self.diffusion.diff_sample(concat_vp)    #[B*T, C, H, W] [80, 512, 14, 14]
                temp_feat = self.kr2(temp_feat) #[B*T, C, H, W] [80, 1024, 14, 14]

                temp_audio, temp_visual = torch.chunk(temp_feat, chunks=2, dim=1)

                temp_audio = temp_audio.view(B,T, C, H, W)
                temp_visual = temp_visual.view(B,T, C, H, W)
                temp_audio = temp_audio.select(3,0).select(3,0)
                
                out_qa, out_match_posi,out_match_nega = self.origin_avqa(origin_audio, temp_visual, visual_nega, qst_feature, xq) 
            elif self.missing_situation == 'both':
                # print("Not missing situation")
                origin_audio = F.relu(self.fc_a1(audio))
                origin_audio = self.fc_a2(origin_audio) 
                diff_for_audio = origin_audio.view(B*T, C)
                diff_for_audio = diff_for_audio.unsqueeze(-1)
                diff_for_audio = diff_for_audio.unsqueeze(-1)
                diff_for_audio = diff_for_audio.repeat(1,1,H,W)
                diff_for_audio = diff_for_audio.view(B*T, C, H, W)
                diff_for_visual = visual_posi.view(B*T, C, H, W)  #(B, T, C, H, W)
                concat_real = torch.cat((diff_for_audio, diff_for_visual), dim=1)
                concat_real = self.kr(concat_real)
                temp_feat = self.diffusion.diff_sample(concat_real)
                temp_feat = self.kr2(temp_feat)
                temp_audio, temp_visual = torch.chunk(temp_feat, chunks=2, dim=1)
                temp_audio = temp_audio.view(B,T, C, H, W)
                temp_visual = temp_visual.view(B,T, C, H, W)
                temp_audio = temp_audio.select(3,0).select(3,0)

                out_qa, out_match_posi,out_match_nega = self.origin_avqa(origin_audio, visual_posi, visual_nega, qst_feature, xq)


            else:
                raise ValueError('Set missing_situation')
        
        
        
            return out_qa, out_match_posi,out_match_nega """
        

