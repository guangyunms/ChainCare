import torch.nn as nn
import torch.nn.functional as F
import torch.nn
from pyhealth.medcode import InnerMap
from torch.nn.utils.rnn import unpack_sequence


def aggregate_tensors(tensors, device):
    """
    将多个张量聚合到最大长度
    :param tensors: list of tensors
    :return: 聚合后的张量, 每个batch的长度
    """
    max_len = max([x.size(1) for x in tensors])
    padded_inputs = []
    lengths = []

    for x in tensors:
        lengths.append(x.size(0))
        padding = torch.zeros(x.size(0), max_len - x.size(1), x.size(2)).to(device)
        padded_x = torch.cat((x, padding), dim=1)
        padded_inputs.append(padded_x)

    aggregated_tensor = torch.cat(padded_inputs, dim=0)
    return aggregated_tensor, lengths

def aggregate_tensors_moniter_event(tensors, device):
    """
    将多个张量聚合到最大长度
    :param tensors: list of tensors
    :return: 聚合后的张量, 每个batch的长度
    """
    max_len_moniter = max([x.size(1) for x in tensors])
    max_len_event = max([x.size(2) for x in tensors])
    padded_inputs = []
    lengths = []

    for x in tensors:
        lengths.append(x.size(0))
        padding_event = torch.zeros(x.size(0), x.size(1), max_len_event - x.size(2), x.size(3)).to(device)
        padded_event = torch.cat((x, padding_event), dim=2)
        padding_moniter = torch.zeros(padded_event.size(0), max_len_moniter - padded_event.size(1), padded_event.size(2), padded_event.size(3)).to(device)
        padded_moniter = torch.cat((padded_event, padding_moniter), dim=1)
        padded_inputs.append(padded_moniter)

    aggregated_tensor = torch.cat(padded_inputs, dim=0)
    return aggregated_tensor, lengths


def split_tensor(tensor, lengths, max_len):
    """
    将聚合的张量拆分为原始形状
    :param tensor: 聚合的张量
    :param lengths: 每个batch的长度
    :param max_len: 最大长度
    :return: 拆分后的张量列表
    """
    index = 0
    outputs = []

    for length in lengths:
        output_tensor = tensor[index:index + length]
        outputs.append(output_tensor)
        index += length

    outputs = [x[:, :max_len, :] for x in outputs]
    return outputs


def extract_and_transpose(tensor_list):
    """
    提取每个张量的最后一个序列并转置
    :param tensor_list: list of tensors
    :return: 处理后的张量列表
    """
    processed_tensors = []
    for tensor in tensor_list:
        last_seq = tensor[:, -1:, :]  # 提取最后一个序列
        transposed_seq = last_seq.transpose(0, 1)  # 转置
        processed_tensors.append(transposed_seq)
    return processed_tensors


class TimeSpace(nn.Module):
    def __init__(
            self,
            Tokenizers_visit_event,
            Tokenizers_monitor_event,
            output_size,
            device,
            embedding_dim=128,
            dropout=0.7,
            trans_num_heads=4,
            trans_num_layers=4
    ):
        super(TimeSpace, self).__init__()
        self.embedding_dim = embedding_dim
        self.visit_event_token = Tokenizers_visit_event
        self.monitor_event_token = Tokenizers_monitor_event

        self.feature_visit_evnet_keys = Tokenizers_visit_event.keys()
        self.feature_monitor_event_keys = Tokenizers_monitor_event.keys()
        self.dropout = torch.nn.Dropout(p=dropout)

        self.device = device

        self.embeddings = nn.ModuleDict()
        # 为每种event（包含monitor和visit）添加一种嵌入
        for feature_key in self.feature_visit_evnet_keys:
            tokenizer = self.visit_event_token[feature_key]
            self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

        for feature_key in self.feature_monitor_event_keys:
            tokenizer = self.monitor_event_token[feature_key]
            self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

        self.visit_gru = nn.ModuleDict()
        # 为每种visit_event添加一种gru
        for feature_key in self.feature_visit_evnet_keys:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        for feature_key in self.feature_monitor_event_keys:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        for feature_key in ['weight', 'age']:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

        self.monitor_gru = nn.ModuleDict()
        # 为每种monitor_event添加一种gru
        for feature_key in self.feature_monitor_event_keys:
            self.monitor_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)


        self.num_heads = trans_num_heads         # 注意力头数
        self.num_layers = trans_num_layers        # Transformer编码器的层数
        self.hidden_dim = 128      # 前馈神经网络的隐藏层维度
        dropout_prob = 0.7    # Dropout概率

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=dropout_prob,
            batch_first=True
        )

        self.transformer = torch.nn.ModuleList([nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers) for i in range(2)])

        self.SGE_weight = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.TGE_weight = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

        self.fc_age = nn.Linear(1, self.embedding_dim)
        self.fc_weight = nn.Linear(1, self.embedding_dim)
        # self.fc_inj_amt = nn.Linear(1, self.embedding_dim)

        item_num = int(len(Tokenizers_monitor_event.keys()) / 2) + 3 + 1
        self.fc_patient = nn.Sequential(
            torch.nn.ReLU(),
            nn.Linear(item_num * self.embedding_dim, output_size)
        )

        # torch.nn.init.xavier_uniform_(self.SGE_weight)
        # torch.nn.init.xavier_uniform_(self.TGE_weight)

    def forward(self, batch_data):

        batch_size = len(batch_data['visit_id'])
        patient_emb_list = []

        """处理lab, inj"""
        feature_paris = list(zip(*[iter(self.feature_monitor_event_keys)] * 2))
        # 迭代处理每一对
        for feature_key1, feature_key2 in feature_paris[:2]:
            monitor_emb_list = []
            # 先聚合monitor层面，生成batch_size个病人的多次就诊的表征，batch_size * (1, visit, embedding)
            for patient in range(batch_size):
                x1 = self.monitor_event_token[feature_key1].batch_encode_3d(
                    batch_data[feature_key1][patient], max_length=(400, 1024)
                )
                x1 = torch.tensor(x1, dtype=torch.long, device=self.device)
                x2 = self.monitor_event_token[feature_key2].batch_encode_3d(
                    batch_data[feature_key2][patient], max_length=(400, 1024)
                )
                x2 = torch.tensor(x2, dtype=torch.long, device=self.device)
                # (visit, monitor, event)

                x1 = self.dropout(self.embeddings[feature_key1](x1))
                x2 = self.dropout(self.embeddings[feature_key2](x2))
                # (visit, monitor, event, embedding_dim)

                x = torch.mul(x1, x2)
                # (visit, monitor, event, embedding_dim)

                x = torch.sum(x, dim=2)
                # (visit, monitor, embedding_dim)

                monitor_emb_list.append(x)

            # 聚合多次的monitor
            aggregated_monitor_tensor, lengths = aggregate_tensors(monitor_emb_list, self.device)
            # (patient * visit, monitor, embedding_dim) 这里不是乘法，而是将多个visit累加

            output, hidden = self.monitor_gru[feature_key1](aggregated_monitor_tensor)
            # output: (patient * visit, monitor, embedding_dim), hidden:(1, patient * visit, embedding_dim)

            # 拆分gru的输出
            max_len = max([x.size(1) for x in monitor_emb_list])
            split_outputs = split_tensor(output, lengths, max_len)

            # 提取最后一个序列并转置
            visit_emb_list = extract_and_transpose(split_outputs)
            # list[batch * (1,visit,dim)]

            # 开始搞visit层面的
            aggregated_visit_tensor, lengths = aggregate_tensors(visit_emb_list, self.device)

            output, hidden = self.visit_gru[feature_key1](aggregated_visit_tensor)
            # output:(patient, visit, embedding_dim), hidden:(1, patient, embedding_dim)

            patient_emb_list.append(hidden.squeeze(dim=0))
            # (patient, event)

        del x, x1, x2
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # 寻找时空关系，先从空间上找，再从时间上找
        for feature_key1, feature_key2 in [feature_paris[3]]:
            group_emb_list = []
            for patient in range(batch_size):
                x1 = self.monitor_event_token[feature_key1].batch_encode_3d(
                    batch_data[feature_key1][patient], max_length=(400, 1024)
                )
                x1 = torch.tensor(x1, dtype=torch.long, device=self.device)
                x1 = self.dropout(self.embeddings[feature_key1](x1))
                # (visit, monitor, event, embedding_dim)

                x2 = self.monitor_event_token[feature_key2].batch_encode_3d(
                    batch_data[feature_key2][patient], max_length=(400, 1024)
                )
                x2 = torch.tensor(x2, dtype=torch.long, device=self.device)
                x2 = self.dropout(self.embeddings[feature_key2](x2))

                x = torch.mul(x1, x2)
                # (visit, monitor, event, embedding_dim)

                x1 = torch.ones(1)
                x2 = torch.ones(1)

                group_emb_list.append(x)

            # group_emb_list: (batch, visit, monitor, event, embedding_dim)
            aggregated_monitor_tensor, lengths = aggregate_tensors_moniter_event(group_emb_list, self.device)
            # aggregated_monitor_tensor: (batch * visit, monitor, event, embedding_dim) 这里不是乘法，而是将多个visit累加

            bv, monitor_num, event_num, embedding = aggregated_monitor_tensor.shape

            event_tensor_reshaped = aggregated_monitor_tensor.view(bv * monitor_num, event_num, self.embedding_dim)
            
            del aggregated_monitor_tensor
            
            # 使用Transformer进行交互信息提取
            event_transformer_output = self.transformer[0](event_tensor_reshaped)
            # (bv * monitor_count, event_count, embedding_dim)

            # GE模块：SoftMax(transformer_output · W)，W是可学习的权重
            SGE_output = torch.matmul(event_transformer_output, self.SGE_weight)

            del event_transformer_output
            # (bv * monitor_count, event_count, embedding_dim)
            SGE_output = torch.softmax(SGE_output, dim=-1)  # 对最后一维做SoftMax

            # 使用GE输出加权transformer输出
            spatial_information_output = self.transformer[0](event_tensor_reshaped + SGE_output)

            del event_tensor_reshaped, SGE_output

            # 将结果reshape回原始的形状(bv, monitor_count, event_count, embedding_dim)
            aggregated_monitor_tensor_2 = spatial_information_output.view(bv, monitor_num, event_num, self.embedding_dim)

            # 现在将其降维到(bv, monitor_count, embedding_dim)，只保留时间信息
            reduced_emb_list = aggregated_monitor_tensor_2.sum(dim=2)

            del aggregated_monitor_tensor_2

            reduced_emb_list_output = self.transformer[1](reduced_emb_list)
            TGE_output = torch.matmul(reduced_emb_list_output, self.TGE_weight)

            del reduced_emb_list_output

            TGE_output = torch.softmax(TGE_output, dim=-1)  # 对最后一维做SoftMax

            # 现在将其降维到(bv, monitor_count, embedding_dim)，只保留时间信息
            reduced_emb_list = self.transformer[1](reduced_emb_list + TGE_output)

            output, hidden = self.monitor_gru[feature_key1](reduced_emb_list)
            # output: (patient * visit, monitor, embedding_dim), hidden:(1, patient * visit, embedding_dim)

            del reduced_emb_list, TGE_output

            # 拆分gru的输出
            max_len = max([x.size(1) for x in group_emb_list])
            split_outputs = split_tensor(output, lengths, max_len)

            # 提取最后一个序列并转置
            visit_emb_list = extract_and_transpose(split_outputs)
            # list[batch * (1,visit,dim)]

            # 开始搞visit层面的
            aggregated_visit_tensor, lengths = aggregate_tensors(visit_emb_list, self.device)

            output, hidden = self.visit_gru[feature_key1](aggregated_visit_tensor)
            # output:(patient, visit, embedding_dim), hidden:(1, patient, embedding_dim)

            patient_emb_list.append(hidden.squeeze(dim=0))
        
        del x, x1, x2
        torch.cuda.empty_cache()

        for feature_key1, feature_key2 in [feature_paris[2]]:
            monitor_emb_list1 = []
            monitor_emb_list2 = []
            for patient in range(batch_size):
                x1 = self.monitor_event_token[feature_key1].batch_encode_3d(
                    batch_data[feature_key1][patient], max_length=(450, 1024) # 有出现过417的
                )
                x1 = torch.tensor(x1, dtype=torch.long, device=self.device)
                x1 = self.dropout(self.embeddings[feature_key1](x1))
                # (visit, monitor, event, embedding_dim)

                x2 = self.monitor_event_token[feature_key2].batch_encode_3d(
                    batch_data[feature_key2][patient], max_length=(450, 1024) # 有出现过417的
                )
                x2 = torch.tensor(x2, dtype=torch.long, device=self.device)
                x2 = self.dropout(self.embeddings[feature_key2](x2))
                # (visit, monitor, event, embedding_dim)

                x = torch.mul(x1, x2)
                # (visit, monitor, event, embedding_dim)

                del x1, x2

                x = torch.sum(x, dim=2)

                reordered_tensor = x.clone().zero_()
                time_index = batch_data['lab_inj_time_index'][patient]

                assert x.size(0) == len(time_index)

                for i in range(x.size(0)):
                    block = x[i]
                    reordered_block = torch.cat((block[time_index[i]], block[len(time_index[i]):]), dim=0)
                    reordered_tensor[i] = reordered_block

                reordered_tensor2 = x.clone().zero_()
                time_index = batch_data['ing_lab_time_index'][patient]

                assert x.size(0) == len(time_index)

                for i in range(x.size(0)):
                    block = x[i]
                    reordered_block = torch.cat((block[time_index[i]], block[len(time_index[i]):]), dim=0)
                    reordered_tensor2[i] = reordered_block

                monitor_emb_list1.append(reordered_tensor)
                monitor_emb_list2.append(reordered_tensor2)

            for monitor_emb_list, feature_key in [(monitor_emb_list1, feature_key1), (monitor_emb_list2, feature_key2)]:

                # 聚合多次的monitor
                aggregated_monitor_tensor, lengths = aggregate_tensors(monitor_emb_list, self.device)
                # (patient * visit, monitor, embedding_dim) 这里不是乘法，而是将多个visit累加

                output, hidden = self.monitor_gru[feature_key](aggregated_monitor_tensor)
                # output: (patient * visit, monitor, embedding_dim), hidden:(1, patient * visit, embedding_dim)

                # 拆分gru的输出
                max_len = max([x.size(1) for x in monitor_emb_list])
                split_outputs = split_tensor(output, lengths, max_len)

                # 提取最后一个序列并转置
                visit_emb_list = extract_and_transpose(split_outputs)
                # list[batch * (1,visit,dim)]

                # 开始搞visit层面的
                aggregated_visit_tensor, lengths = aggregate_tensors(visit_emb_list, self.device)

                output, hidden = self.visit_gru[feature_key](aggregated_visit_tensor)
                # output:(patient, visit, embedding_dim), hidden:(1, patient, embedding_dim)

                patient_emb_list.append(hidden.squeeze(dim=0))

        """处理cond, proc, drug"""
        for feature_key in self.feature_visit_evnet_keys:
            x = self.visit_event_token[feature_key].batch_encode_3d(
                batch_data[feature_key]
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, event)

            x = self.dropout(self.embeddings[feature_key](x))
            # (patient, visit, event, embedding_dim)

            x = torch.sum(x, dim=2)
            # (patient, visit, embedding_dim)

            output, hidden = self.visit_gru[feature_key](x)
            # output:(patient, visit, embedding_dim), hidden:(1, patient, embedding_dim)

            patient_emb_list.append(hidden.squeeze(dim=0))


        patient_emb = torch.cat(patient_emb_list, dim=-1)
        # (patient, 6 * embedding_dim)

        logits = self.fc_patient(patient_emb)
        # (patient, label_size)
        return logits
