from __future__ import print_function
import sys 
import os
sys.path.append(os.getcwd()) 
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from net_grd_avst.dataloader_avst import *
from net_grd_avst.net_avst import AVQA_Fusion_Net
import ast
import json
import numpy as np
import pdb
import torch.nn.functional as F
from tqdm.auto import tqdm

import warnings
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
import time

import sys


writer = SummaryWriter('runs/net_avst/'+'concat_mean_pr'+TIMESTAMP)

print("\n--------------- Audio-Visual Spatial-Temporal Model --------------- \n")

def batch_organize(out_match_posi,out_match_nega):
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


def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_qa = 0
    correct_qa = 0
    for batch_idx, sample in tqdm(enumerate(train_loader)):
        audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

        optimizer.zero_grad()
        # kyuri add diff_loss, ref_loss, audio_feat_ref
        avqa_output, diff_loss, memory_loss, temp_loss = model(audio, visual_posi,visual_nega, question)  
        
        # visiual
        out_match,match_label=batch_organize( avqa_output["out_match_posi"], avqa_output["out_match_nega"])
        out_match,match_label = out_match.type(torch.FloatTensor).cuda(), match_label.type(torch.LongTensor).cuda()

        out_match_ap,match_label_ap=batch_organize(avqa_output["visual_match_posi_ap"], avqa_output["visual_match_nega_ap"])
        out_match_ap,match_label_ap = out_match_ap.type(torch.FloatTensor).cuda(), match_label_ap.type(torch.LongTensor).cuda()

        out_match_vp,match_label_vp=batch_organize(avqa_output["visual_match_posi_vp"], avqa_output["visual_match_nega_vp"])
        out_match_vp,match_label_vp = out_match_vp.type(torch.FloatTensor).cuda(), match_label_vp.type(torch.LongTensor).cuda()
    
        # output.clamp_(min=1e-7, max=1 - 1e-7)
        loss_match=criterion(out_match,match_label) # out_match(output of model)와 match_label(정답 label) 사이의 손실 계산. kyuri
        loss_qa = criterion(avqa_output["out_qa"], target)
        origin_loss = loss_qa + 0.5*loss_match

        loss_match_ap=criterion(out_match_ap,match_label_ap) # out_match(output of model)와 match_label(정답 label) 사이의 손실 계산. kyuri
        loss_qa_ap = criterion(avqa_output["out_qa_ap"], target)
        origin_loss_ap = loss_qa_ap + 0.5*loss_match_ap

        loss_match_vp=criterion(out_match_vp,match_label_vp) # out_match(output of model)와 match_label(정답 label) 사이의 손실 계산. kyuri
        loss_qa_vp = criterion(avqa_output["out_qa_vp"], target)
        origin_loss_vp = loss_qa_vp + 0.5*loss_match_vp

        # origin_loss.mean()
        # origin_loss2.mean()
        # memory_loss.mean()
        # temp_loss.mean()

        """ if args.step=="step2":
            loss = origin_loss + origin_loss_ap + origin_loss_vp + diff_loss + temp_loss
        else:
            loss = origin_loss + origin_loss_ap + origin_loss_vp + diff_loss + memory_loss """
        
        loss = origin_loss + origin_loss_ap + origin_loss_vp + diff_loss + memory_loss + temp_loss

        writer.add_scalar('run/match',loss_match.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/match2',loss_match_ap.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/match3',loss_match_vp.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/qa_test',loss_qa.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/qa_test2',loss_qa_ap.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/qa_test3',loss_qa_vp.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/both',loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/diff_loss',diff_loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/memory_loss',memory_loss.item(), epoch * len(train_loader) + batch_idx)

        loss.backward()
        optimizer.step()
        #print('batch_idx: ', batch_idx)
        if batch_idx % args.log_interval == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Total Loss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print(f"iterations: {batch_idx} / {len(train_loader)}")
            print('diff_loss: {:.2f}'.format(diff_loss.item()))
            print('memory_loss: {:.2f}'.format(memory_loss.item()))
            print('origin_loss: {:.2f}'.format(origin_loss.item()))
            print('temp_loss: {:.2f}'.format(temp_loss.item()))
            print('origin_loss_ap: {:.2f}'.format(origin_loss_ap.item()))
            print('origin_loss_vp: {:.2f}'.format(origin_loss_vp.item()))
            print('===============AVST_ours_no step============================')
        



def eval(model, val_loader,epoch):
    model.eval()
    total_qa = 0
    total_match=0
    correct_qa = 0
    correct_match=0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')
            
            # preds_qa, out_match_posi,out_match_nega = model(audio, visual_posi,visual_nega, question)
            preds_qa, out_match_posi,out_match_nega = model(audio, visual_posi, visual_nega, question)
            # if model.training:
            #     preds_qa = results
            # else:
            #     preds_qa = results
            _, predicted = torch.max(preds_qa.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{}]\t '.format(epoch, batch_idx * len(audio), len(val_loader.dataset)))


    print('Accuracy qa: %.2f %%' % (100 * correct_qa / total_qa))
    writer.add_scalar('metric/acc_qa',100 * correct_qa / total_qa, epoch)

    return 100 * correct_qa / total_qa

def test(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open('./data/json_update/avqa-test.json', 'r'))
    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    inference_times = []

    with torch.no_grad():

        for batch_idx, sample in enumerate(val_loader):

            batch_start_time = time.time()
            
            if batch_idx % 1000 == 0:
                print(batch_idx, '/', val_loader.__len__())

            audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')


            preds_qa,_,_ = model(audio, visual_posi,visual_nega, question)
            preds = preds_qa
            _, predicted = torch.max(preds.data, 1)

            total += preds.size(0)
            correct += (predicted == target).sum().item()

            x = samples[batch_idx]
            type =ast.literal_eval(x['type'])
            if type[0] == 'Audio':
                if type[1] == 'Counting':
                    A_count.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    A_cmp.append((predicted == target).sum().item())
            elif type[0] == 'Visual':
                if type[1] == 'Counting':
                    V_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    V_loc.append((predicted == target).sum().item())
            elif type[0] == 'Audio-Visual':
                if type[1] == 'Existential':
                    AV_ext.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    AV_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    AV_loc.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    AV_cmp.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    AV_temp.append((predicted == target).sum().item())
        
            batch_end_time = time.time()
            inference_time = batch_end_time - batch_start_time
            inference_times.append(inference_time)

            # 각 iteration의 inference time 출력
            print(f"Iteration {batch_idx + 1} inference time: {inference_time:.6f} seconds")
            
            # 100 iteration마다 평균 inference time 출력
            if (batch_idx + 1) % 100 == 0:
                avg_inference_time = sum(inference_times[-100:]) / 100
                print(f"Average inference time for last 100 batches: {avg_inference_time:.6f} seconds")

        total_end_time = time.time()
        total_time = total_end_time - batch_start_time
        average_time = total_time / len(val_loader)
        print(f"Average evaluation time per image: {average_time:.6f} seconds.")
    
    print('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count)/len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    print('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    print('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    print('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    print('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    print('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    print('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    print('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))

    print('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                   +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))

    print('Overall Accuracy: %.2f %%' % (
            100 * correct / total))

    return 100 * correct / total

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')

    parser.add_argument(
        "--audio_dir", type=str, default='./data/feats/vggish', help="audio dir")
    # parser.add_argument(
    #     "--video_dir", type=str, default='/home/guangyao_li/dataset/avqa/avqa-frames-1fps', help="video dir")
    parser.add_argument(
        "--video_res14x14_dir", type=str, default='./data/feats/res18_14x14', help="res14x14 dir")
    
    parser.add_argument(
        "--label_train", type=str, default="./data/json_update/avqa-train.json", help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default="./data/json_update/avqa-val.json", help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default="./data/json_update/avqa-test.json", help="test csv file")
    parser.add_argument(
        '--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 16)') #princess default=64
    parser.add_argument(
        '--epochs', type=int, default=80, metavar='N', help='number of epochs to train (default: 60)') #princess default=80
    parser.add_argument(
        '--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='AVQA_Fusion_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='./net_grd_avst/avst_models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='avst', help="save model name")
    # kyuri
    parser.add_argument('--memory_size', type=int, default=75,
                    help='memory slot size')
    parser.add_argument('--missing_situation', type=str, default="visual",
                    help='missing situation for test')
    parser.add_argument(
        "--test_path", type=str, default="./net_grd_avst/avst_models/avst/best.pt", help="test path")
    parser.add_argument(
        "--pretrained_path", type=str, default="./net_grd_avst/avst_models/avst/best.pt", help="pretrained_path")
 
    parser.add_argument(
        "--log_file", type=str, default='stdout.txt', help="save model name")
    parser.add_argument(
        "--time_step", type=int, default=5, help="diffusion time step")
#    parser.add_argument(
#        '--gpu', type=str, default='1,2,3,4', help='gpu device number')


    args = parser.parse_args()
#    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(args.seed)
    sys.stdout = open('logs/'+args.log_file, 'w')


    if args.model == 'AVQA_Fusion_Net':
        model = AVQA_Fusion_Net(args)
        model = nn.DataParallel(model)
        model = model.to('cuda')
        model.module.set_missing_situation(args.missing_situation)
    else:
        raise ('not recognized')

    if args.mode == 'train':
        model.load_state_dict(torch.load(args.pretrained_path), strict=False)
        """ if args.step=="step2":
            # freeze the parameters kyuri
            for key in model.state_dict().keys():
                 if not "diffusion" in key and not "unet" in key:  # diffusion layer만 학습
                    model.state_dict()[key].requires_grad = False """
                    


        train_dataset = AVQA_dataset(label=args.label_train, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                    transform=transforms.Compose([ToTensor()]), mode_flag='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dataset = AVQA_dataset(label=args.label_val, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                    transform=transforms.Compose([ToTensor()]), mode_flag='val')
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        print(train_dataset.__len__(), val_dataset.__len__())

        # # ===================================== load pretrained model ===============================================
        # ####### concat model
        # pretrained_file = "grounding_gen/models_grounding_gen/main_grounding_gen_best.pt"
        # checkpoint = torch.load(pretrained_file)
        # print("\n-------------- loading pretrained models --------------")
        # model_dict = model.state_dict()
        # # tmp = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias','module.fc_gl.weight','module.fc_gl.bias','module.fc1.weight', 'module.fc1.bias','module.fc2.weight', 'module.fc2.bias','module.fc3.weight', 'module.fc3.bias','module.fc4.weight', 'module.fc4.bias']
        # tmp = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias','module.fc_gl.weight','module.fc_gl.bias','module.fc1.weight', 'module.fc1.bias','module.fc2.weight', 'module.fc2.bias','module.fc3.weight', 'module.fc3.bias','module.fc4.weight', 'module.fc4.bias','module.conv1.weight', 'module.conv1.bias','module.kr_conv.weight','module.kr_conv.bias']
        # tmp2 = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias']
        # pretrained_dict1 = {k: v for k, v in checkpoint.items() if k in tmp}
        # pretrained_dict2 = {str(k).split('.')[0]+'.'+str(k).split('.')[1]+'_pure.'+str(k).split('.')[-1]: v for k, v in checkpoint.items() if k in tmp2}

        # model_dict.update(pretrained_dict1) # Use the pre-trained model's parameters to update the model.
        # model_dict.update(pretrained_dict2) # Use the pre-trained model's parameters to update the model.
        # model.load_state_dict(model_dict)


        # print("\n-------------- load pretrained models --------------")

        # # ===================================== load pretrained model ===============================================

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) #kyuri scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_F = 0
        # if not os.path.exists(args.model_save_dir + args.checkpoint):
        os.makedirs(args.model_save_dir + args.checkpoint, exist_ok=True)
        
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            F = eval(model, val_loader, epoch)

            # kyuri weight 저장
            if (epoch % 10) == 0:
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + '/' + "weight_epoch_{}.pt".format(epoch))

            if F >= best_F:
                best_F = F
                print('The current epoch is the best: ' + str(epoch) + 'epoch')
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + '/' + "best" +".pt")
            
            print('Best Accuracy: {:.2f}%'.format(best_F))  # kyuri best accuracy 뽑기
                
    else:
        #kyuri
        # model.load_state_dict(torch.load(args.test_path), strict=False)
        model.load_state_dict(torch.load(args.test_path), strict=True)
        model = nn.DataParallel(model)  # gpu 여러개 쓸 수 있도록 parallel 작업 시키도록 모델을 바꾸는 거임. 그래서 로그처럼 module이라는게 생기고 그래서 dataparallel하기 전에 load를 하면 문제 없음.
        test_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, video_res14x14_dir=args.video_res14x14_dir,
                                   transform=transforms.Compose([ToTensor()]), mode_flag='test')
        print(test_dataset.__len__())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        # model.load_state_dict(torch.load(args.test_path), strict=False)
        # model.load_state_dict(torch.load(args.test_path))


        test(model, test_loader)
    sys.stdout.close()



if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    main()
