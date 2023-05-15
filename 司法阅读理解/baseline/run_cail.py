import argparse
from os.path import join
from tqdm import tqdm
from transformers import BertModel,RobertaModel
from transformers import BertConfig as BC

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
# from model.modeling import *
from model.GNN import *
from tools.utils import convert_to_tokens
from tools.data_iterator_pack import IGNORE_INDEX
import numpy as np
import queue
import random
from config import set_config
from tools.data_helper import DataHelper
from data_process import InputFeatures,Example
try:
    from apex import amp
except Exception:
    print('Apex not imoport!')


import torch
from torch import nn


def set_seed(args):
    # 设置Python的随机数种子
    random.seed(args.seed)
    # 设置NumPy的随机数种子  
    np.random.seed(args.seed)  
    # 设置PyTorch的随机数种子
    torch.manual_seed(args.seed)  
    # 设置所有GPU的随机数种子
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  


def dispatch(context_encoding, context_mask, batch, device):
    # 将"context_encoding"张量移动到指定设备上
    batch['context_encoding'] = context_encoding.cuda(device)  
    # 将"context_mask"张量转换为浮点型并移动到指定设备上
    batch['context_mask'] = context_mask.float().cuda(device)  
    return batch

def compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position):
    # 计算起始位置和结束位置的损失
    loss1 = criterion(start_logits, batch['y1']) + criterion(end_logits, batch['y2']) 
    # 使用系数计算类型的损失
    loss2 = args.type_lambda * criterion(type_logits, batch['q_type'])  
    # 统计批次中的句子数量
    sent_num_in_batch = batch["start_mapping"].sum()  
    # 使用系数计算支持事实的损失，并通过句子数量进行归一化
    loss3 = args.sp_lambda * sp_loss_fct(sp_logits.view(-1), batch['is_support'].float().view(-1)).sum() / sent_num_in_batch  
    # 计算总损失，将各个损失相加
    loss = loss1 + loss2 + loss3  
    return loss, loss1, loss2, loss3



import json

@torch.no_grad()
def predict(model, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=False):
    '''
    使用训练好的模型对数据进行预测，并将预测结果保存到文件中。
    :model: 训练好的模型
    :dataloader: 数据加载器，用于遍历预测数据
    :example_dict: 示例字典，包含输入样本的相关信息
    :feature_dict: 特征字典，包含输入特征的相关信息
    :prediction_file: 预测结果保存的文件路径
    :need_sp_logit_file: 是否需要保存支持事实的logit分数文件，默认为False
    '''
    # 设置模型为评估模式，不进行梯度计算
    model.eval()
    # 存储答案的字典
    answer_dict = {}
    # 存储支持事实的字典
    sp_dict = {}
    # 刷新数据加载器，重置迭代状态
    dataloader.refresh()
    # 存储测试损失的列表
    total_test_loss = [0] * 5

    # 遍历数据加载器中的批次数据
    for batch in tqdm(dataloader):
        # 将上下文掩码转换为浮点型
        batch['context_mask'] = batch['context_mask'].float()
        # 使用模型进行预测，得到预测结果
        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(batch)
        # 计算损失
        loss_list = compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position)

        for i, l in enumerate(loss_list):
            if not isinstance(l, int):
                # 累计测试损失
                total_test_loss[i] += l.item()

        # 将预测结果转换为文本形式
        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'], start_position.data.cpu().numpy().tolist(),
                                         end_position.data.cpu().numpy().tolist(), np.argmax(type_logits.data.cpu().numpy(), 1))
        # 更新答案字典
        answer_dict.update(answer_dict_)

        # 对支持事实的logits进行sigmoid激活，并转换为NumPy数组
        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            # 当前样本的支持事实预测列表
            cur_sp_pred = []
            # 当前样本的ID
            cur_id = batch['ids'][i]

            # 用于保存支持事实logit的输出
            cur_sp_logit_pred = []  # for sp logit output
            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if need_sp_logit_file:
                    temp_title, temp_id = example_dict[cur_id].sent_names[j]
                    # 保存支持事实logit的分数
                    cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
                if predict_support_np[i, j] > args.sp_threshold:
                    # 将预测为支持事实的句子添加到当前样本的支持事实预测列表中
                    cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            # 更新支持事实字典
            sp_dict.update({cur_id: cur_sp_pred})

    new_answer_dict={}
    for key,value in answer_dict.items():
        # 去除答案中的空格并更新到新的答案字典中
        new_answer_dict[key]=value.replace(" ","")
    # 构建预测结果字典
    prediction = {'answer': new_answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w',encoding='utf8') as f:
        # 将预测结果字典保存到文件中
        json.dump(prediction, f,indent=4,ensure_ascii=False)

    for i, l in enumerate(total_test_loss):
        # 打印每个测试损失项的平均值
        print("Test Loss{}: {}".format(i, l / len(dataloader)))
    # 将前三个测试损失项相加，除以数据加载器中批次的数量，得到总的测试损失值
    test_loss_record.append(sum(total_test_loss[:3]) / len(dataloader))


def train_epoch(data_loader, model, predict_during_train=False):
    '''
    对一个训练 epoch 进行训练，并可选择在训练过程中进行预测。
    :data_loader: 数据加载器，用于遍历训练数据
    :model: 训练的模型
    :predict_during_train: 是否在训练过程中进行预测，默认为 False
    '''
    # 设置模型为训练模式
    model.train()
    # 创建进度条
    pbar = tqdm(total=len(data_loader))
    # 获取 epoch 的长度（批次数）
    epoch_len = len(data_loader)
    # 记录当前的步数
    step_count = 0
    # 指定进行预测的步数间隔（将 epoch 分为 5 个部分）
    predict_step = epoch_len // 5

    # 当数据加载器不为空时循环
    while not data_loader.empty():
        # 增加步数计数
        step_count += 1
        # 获取当前批次的数据
        batch = next(iter(data_loader))
        # 将上下文掩码转换为浮点型
        batch['context_mask'] = batch['context_mask'].float()
        # 对当前批次的数据进行训练
        train_batch(model, batch)
        # 释放批次数据的内存
        del batch

        if predict_during_train and (step_count % predict_step == 0):
            # 在训练过程中进行预测，并保存预测结果和模型参数
            predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
                     join(args.prediction_path, 'pred_seed_{}_epoch_{}_{}.json'.format(args.seed, epc, step_count)))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), join(args.checkpoint_path, "ckpt_seed_{}_epoch_{}_{}.pth".format(args.seed, epc, step_count)))
            # 恢复模型为训练模式
            model.train()
        # 更新进度条
        pbar.update(1)

    # 在 epoch 结束时进行最后一次预测，并保存预测结果和模型参数
    predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
             join(args.prediction_path, 'pred_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)))
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), join(args.checkpoint_path, "ckpt_seed_{}_epoch_{}_99999.pth".format(args.seed, epc)))


def train_batch(model, batch):
    '''
    对一个训练批次进行训练。
    '''
    # 全局变量
    global global_step, total_train_loss

    # 模型前向传播，获取预测结果
    start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(batch)
    # 计算损失
    loss_list = compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position)
    # 将损失列表转换为可索引的列表
    loss_list = list(loss_list)
    if args.gradient_accumulation_steps > 1:
        # 如果采用梯度累积，则将第一个损失除以梯度累积的步数
        loss_list[0] = loss_list[0] / args.gradient_accumulation_steps
    
    if args.fp16:
        # 使用半精训练时，对损失进行自动缩放并进行反向传播
        with amp.scale_loss(loss_list[0], optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        # 普通精度训练时，对损失进行反向传播
        loss_list[0].backward()

    if (global_step + 1) % args.gradient_accumulation_steps == 0:
        # 更新模型参数
        optimizer.step()
        # 更新学习率
        scheduler.step()
        # 清空梯度
        optimizer.zero_grad()
    
    # 增加全局步数计数器
    global_step += 1

    for i, l in enumerate(loss_list):
        if not isinstance(l, int):
            # 累计损失值
            total_train_loss[i] += l.item()

    if global_step % VERBOSE_STEP == 0:
        print("{} -- In Epoch{}: ".format(args.name, epc))
        for i, l in enumerate(total_train_loss):
            # 打印每个损失项的平均损失值
            print("Avg-LOSS{}/batch/step: {}".format(i, l / VERBOSE_STEP))
        # 重置损失列表
        total_train_loss = [0] * 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = set_config()

    # 重置损失列表
    args.n_gpu = torch.cuda.device_count()

    # 如果args.seed为0，则随机生成一个种子，并调用set_seed函数设置随机种子。
    if args.seed == 0:
        args.seed = random.randint(0, 100)
        set_seed(args)

    # 创建DataHelper对象helper, 启用压缩
    helper = DataHelper(gz=True, config=args)
    args.n_type = helper.n_type  # 2

    # 取数据集：
    Full_Loader = helper.train_loader
    # Subset_Loader = helper.train_sub_loader
    dev_example_dict = helper.dev_example_dict
    dev_feature_dict = helper.dev_feature_dict
    eval_dataset = helper.dev_loader


    # 加载RoBERTa模型的配置文件roberta_config。
    roberta_config = BC.from_pretrained(args.bert_model)
    # 加载Bert模型的编码器encoder。
    encoder = BertModel.from_pretrained(args.bert_model)
    # encoder = RobertaModel.from_pretrained(args.bert_model)
    
    # 同步模型输入DIM为预训练模型的
    args.input_dim=roberta_config.hidden_size
    # 创建BertSupportNet模型对象model，并将编码器和配置参数传递给它。
    model = BertSupportNet(config=args, encoder=encoder)
    # 如果指定了预训练权重args.trained_weight，则加载权重到模型中。
    if args.trained_weight is not None:
        model.load_state_dict(torch.load(args.trained_weight))
    model.to('cuda')

    # 初始化优化器和损失函数
    lr = args.lr
    t_total = len(Full_Loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = 0.1 * t_total
    # 使用AdamW优化器，设置学习率为args.lr，epsilon为1e-8。
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    # 学习率调度器，设置预热步数为总步数的10%。
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    # 使用交叉熵损失函数进行分类任务的损失计算，忽略索引为IGNORE_INDEX的标签。
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)  
    # 使用nn.BCEWithLogitsLoss作为二分类任务的损失函数。
    binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')  
    # 使用nn.BCEWithLogitsLoss作为支持文本的损失函数，设置reduction='none'以便后续计算
    sp_loss_fct = nn.BCEWithLogitsLoss(reduction='none')  

    # 使用半精训练则导入apex库并使用amp.initialize函数对模型和优化器进行混合精度初始化。
    if args.fp16:
        import apex
        apex.amp.register_half_function(torch, "einsum")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # 使用torch.nn.DataParallel将模型并行化，以加速训练。
    model = torch.nn.DataParallel(model)
    model.train()

    # Training
    # 初始化全局步数计数器global_step和当前的训练轮数epc。
    global_step = epc = 0
    # 初始化训练过程中的总损失列表total_train_loss和测试损失记录列表test_loss_record。
    total_train_loss = [0] * 5
    test_loss_record = []
    # 设置训练过程中的详细输出步数VERBOSE_STEP。
    VERBOSE_STEP = args.verbose_step

    while True:
        # 如果epc等于args.epochs，即训练轮数达到设定值，则退出程序。
        if epc == args.epochs:  # 5 + 30
            exit(0)
        epc += 1

        Loader = Full_Loader
        # 刷新训练数据集的数据加载器Loader。
        Loader.refresh()

        # 第二轮开始训练过程中进行预测。
        if epc > 2:
            train_epoch(Loader, model, predict_during_train=True)
        else:
            train_epoch(Loader, model)
