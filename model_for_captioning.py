"""
Adapted from SwinBERT
"""

from utils.lib import *
from model import LAVENDER_Base
import torch.nn.functional as F
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer, BertModel, BertTokenizer
import json
import re

class BERTTextEncoder(T.nn.Module):
    def __init__(
        self,
        hidden_size,
        learnable_length,
        ouput_size,
        score_emb_num,
        bert_model_name="bert-base-uncased",
    ):
        super(BERTTextEncoder, self).__init__()
        # Load configs
        self.score_emb_num = score_emb_num
        self.learnable_length = learnable_length
        # Set text_encoder and learnable prompt embeddings.
        self.learnable_embeddings = T.nn.Parameter(
            T.randn(1, learnable_length, hidden_size)
        )
        # self.learnable_embeddings = torch.nn.Parameter(torch.randn(1, learnable_length * score_emb_num, hidden_size))
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.text_encoder = self.bert_model.encoder
        self.embeddings = self.bert_model.embeddings
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        del self.bert_model


        if self.score_emb_num == 2:
            self.score_groups_word = ["good", "bad"]
        else:
            self.score_groups_word = []
            keypoint_path = "./step_instructions_translated_keys.json"
            with open(keypoint_path, "r") as f:
                file_content = f.read()
                data = json.loads(file_content)
            for action_name, steps_and_instructions in data.items():
                # Create a list to hold the cleaned text for this action
                cleaned_text_for_action = []
                action_name = action_name + ":"
                # Start with the action name itself
                cleaned_text_for_action.append(action_name)

                # Process each step/instruction
                for item in steps_and_instructions:
                    # Find the first colon to locate the start of the actual text
                    colon_index = item.find(":")
                    if colon_index != -1:
                        # Extract the text after the first colon and strip leading/trailing whitespace
                        cleaned_item = item[colon_index + 1 :].strip()
                        if cleaned_item:  # Only add if there's actual text
                            cleaned_text_for_action.append(cleaned_item)
                    else:
                        # If no colon is found (unlikely in your example, but good practice)
                        cleaned_item = item.strip()
                        if cleaned_item:
                            cleaned_text_for_action.append(cleaned_item)

                # Combine the action name and cleaned steps/instructions into a single string
                # You can choose how to separate them, here using a space
                combined_string = " ".join(cleaned_text_for_action)
                self.score_groups_word.append(combined_string)


    def _build_causal_attention_mask(self, bsz, seq_len, dtype):

        mask = T.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(T.tensor(T.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask [100, 1, 29, 29]
        return mask

    def forward(self, x):
        if x == 0:
            x = x
        else:
            self.score_groups_word = x
        input_ids = self.tokenizer(
            self.score_groups_word, padding=True, return_tensors="pt"
        )["input_ids"].cuda()
        input_ids = input_ids[:, 1:-1]  # [100, 11] -> [100, 9]
        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=None
        )  # [100, 9, 512]
        learnable_embeddings = self.learnable_embeddings.expand(
            hidden_states.shape[0], -1, -1
        )  # [1, 20, 512] -> [100, 20, 512]

        hidden_states = T.cat(
            (learnable_embeddings, hidden_states), dim=1
        )  # [100, 29, 512]

        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])

        bsz, seq_len = input_shape  # 100, 9
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len + self.learnable_length, hidden_states.dtype
        ).to(hidden_states.device)
        # expand attention_mask
        # if attention_mask is not None:
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
        encoder_outputs = self.text_encoder(
            hidden_states=hidden_states,  
            attention_mask=None, 
            output_attentions=False,
            output_hidden_states=True,
            return_dict=False,
        )[0]


        return encoder_outputs
        # return hidden_states


class CaptioningLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_smoothing = getattr(config, "label_smoothing", 0)
        self.drop_worst_ratio = getattr(config, "drop_worst_ratio", 0)
        self.drop_worst_after = getattr(config, "drop_worst_after", 0)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction="none")
        self.iter = 0

    def forward(self, logits, target):
        self.iter += 1
        eps = self.label_smoothing
        n_class = logits.size(1)
        one_hot = T.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(logits)
        loss = self.kl(log_prb, one_hot).sum(1)

        if self.drop_worst_ratio > 0 and self.iter > self.drop_worst_after:
            loss, _ = T.topk(
                loss, k=int(loss.shape[0] * (1 - self.drop_worst_ratio)), largest=False
            )

        loss = loss.mean()

        return loss


class CrossAttentionLayer(T.nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        output_attentions,
    ):
        super(CrossAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        self.output_attentions = output_attentions
        q_size = hidden_size
        k_size = hidden_size

        self.query = T.nn.Linear(q_size, self.all_head_size)
        self.key = T.nn.Linear(k_size, self.all_head_size)
        self.value = T.nn.Linear(k_size, self.all_head_size)

        self.layernorm = T.nn.LayerNorm(hidden_size)
        self.feedforward = T.nn.Linear(hidden_size, hidden_size)
        self.dropout = T.nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # (B, L, hidden_size) -> (B, num_heads, L, hidden_size/num_heads)
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, attention_mask=None, head_mask=None):
        v = k  # # q [4, 784, 512] k [4, 100, 512]
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [4, 8, 784, 64]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [4, 8, 100, 64]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [4, 8, 100, 64]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = T.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )  # [4, 8, 784, 100]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = T.nn.Softmax(dim=-1)(attention_scores)  # [4, 8, 784, 100]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = T.matmul(attention_probs, value_layer)  # [4, 8, 784, 64]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [4, 784, 512]

        # Add & Norm
        context_layer = context_layer + q
        context_layer = self.layernorm(context_layer)

        # Feed Forward
        context_layer = self.feedforward(context_layer)

        # Add & Norm
        context_layer = context_layer + q
        context_layer = self.layernorm(context_layer)

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


# create a multi-layers cross attention module
class CrossAttentionModel(T.nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob=0,
        output_attentions=True,
    ):
        super(CrossAttentionModel, self).__init__()
        model_list = []
        for i in range(num_layers):
            output_attentions = True if i == num_layers - 1 else False
            model_list.append(
                CrossAttentionLayer(
                    hidden_size,
                    num_attention_heads,
                    attention_probs_dropout_prob,
                    output_attentions,
                )
            )
        self.model = T.nn.ModuleList(model_list)

    def forward(self, q, k):
        outputs = (q,)  # q [4, 784, 512] k [4, 100, 512]
        for _layer in self.model:
            outputs = _layer(outputs[0], k)
        return outputs


class EFA_Captioning(
    LAVENDER_Base
):  
   

    def __init__(self, args, tokzr, is_decoder=True):
        super().__init__(args, tokzr)
        self.config.is_decoder = is_decoder
        bert = transformers.AutoModelForMaskedLM.from_pretrained(
            self.args.tokenizer, config=self.config
        )
        self.fc_mtm = bert.cls
        del bert
        self.tokenizer = BertTokenizer.from_pretrained(self.args.tokenizer)
        self.clip_text_model_class = BERTTextEncoder(
            hidden_size=768,  # 512
            learnable_length=5,
            ouput_size=512,
            score_emb_num=141,
        )
        self.clip_text_model_score = BERTTextEncoder(
            hidden_size=768,  # 512
            learnable_length=5,
            ouput_size=512,
            score_emb_num=2,
        )
        self.clip_text_model_step = BERTTextEncoder(
            hidden_size=768,  # 512
            learnable_length=5,
            ouput_size=512,
            score_emb_num=24,
        )
        self.task_tok2id = {"vtm": 0, "mc": 1, "oe": 2, "cap": 3}
        self.emb_task = T.nn.Parameter(0.02 * T.randn(10, self.hidden_size))
        self.cap_prompt_txt_L = 0
        # Cross Attention for score
        self.cross_att_1 = CrossAttentionModel(
            num_layers=8,
            hidden_size=self.hidden_size,
            num_attention_heads=8,
            attention_probs_dropout_prob=0,
            output_attentions=True,
        )
        self.cross_att_2 = CrossAttentionModel(
            num_layers=8,
            hidden_size=self.hidden_size,
            num_attention_heads=8,
            attention_probs_dropout_prob=0,
            output_attentions=True,
        )
        # Cross Attention for class
        self.cross_att_3 = CrossAttentionModel(
            num_layers=8,
            hidden_size=self.hidden_size,
            num_attention_heads=8,
            attention_probs_dropout_prob=0,
            output_attentions=True,
        )
        self.cross_att_4 = CrossAttentionModel(
            num_layers=8,
            hidden_size=self.hidden_size,
            num_attention_heads=8,
            attention_probs_dropout_prob=0,
            output_attentions=True,
        )
        self.cross_att_5 = CrossAttentionModel(
            num_layers=8,
            hidden_size=self.hidden_size,
            num_attention_heads=8,
            attention_probs_dropout_prob=0,
            output_attentions=True,
        )
        self.cross_att_6 = CrossAttentionModel(
            num_layers=8,
            hidden_size=self.hidden_size,
            num_attention_heads=8,
            attention_probs_dropout_prob=0,
            output_attentions=True,
        )
        self.fusion_projection = T.nn.Sequential(
            T.nn.Linear(self.hidden_size * 2, self.hidden_size),
            T.nn.ReLU(),
            T.nn.LayerNorm(self.hidden_size),
        )
        keypoint_path = "./step_instructions_translated_keys.json"
        
        with open(keypoint_path, "r") as f:
            file_content = f.read()
            data = json.loads(file_content)
        self.items_list = list(data.items())
        self.weight_for_prompt = T.nn.Parameter(T.ones(self.hidden_size) * 1e-3)
        self.weight_for_vidfeats = T.nn.Parameter(T.ones(self.hidden_size) * 1e-3)
        # 用于计算视频特征处理前后加权的动态门控（类resnet原理）
        self.gate_projection = T.nn.Linear(self.hidden_size * 2, self.hidden_size)
        #用于融合全局和步骤感知两种视频特征的动态门控
        self.fusion_gate_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )

        # 在 forward 函数中融合
        # 使用 sigmoid 函数确保 alpha 的值在 (0, 1) 区间
        
        
        # Regressor for AQA
        self.regressor_aqa_1 = T.nn.Linear(self.hidden_size, 128)
        # # self.regressor_aqa_2 = torch.nn.Linear(128, 64)
        self.regressor_aqa_2 = T.nn.Linear(128, 32)
        # # self.regressor_aqa_3 = torch.nn.Linear(64, 32)
        self.regressor_aqa_3 = T.nn.Linear(32, 1)
        # # self.regressor_aqa_4 = torch.nn.Linear(32, 1)
        self.regressor_aqa_1_class = T.nn.Linear(self.hidden_size, 256)
        # # self.regressor_aqa_2 = torch.nn.Linear(128, 64)
        self.regressor_aqa_2_class = T.nn.Linear(256, 128)
        # # self.regressor_aqa_3 = torch.nn.Linear(64, 32)
        self.regressor_aqa_3_class = T.nn.Linear(128, 141)
        self.use_prompt = 1
        self.use_texthead = 0

    def get_prompt(self, prompt_text=None):
        if prompt_text is None:
            prompt_text = "write a description about the video."
        toks = self.tokzr.tokenize(prompt_text)
        txt = (
            [self.cls_token_id]
            + self.tokzr.convert_tokens_to_ids(toks)
            + [self.sep_token_id]
        )
        mask = [1 if w != self.pad_token_id else w for w in txt]
        mask = T.LongTensor(mask)
        txt = T.LongTensor(txt)
        return txt, mask

    def get_aqa_score(self, vid_feats):
        pred_vid_feats = vid_feats.mean(dim=1)  # (B x 300 x 768)  ->  (B x 768)
        B = pred_vid_feats.shape[0]

        scores_pred = F.relu(self.regressor_aqa_1(pred_vid_feats))
        scores_pred = F.relu(self.regressor_aqa_2(scores_pred))
        scores_pred = self.regressor_aqa_3(scores_pred)

        return scores_pred

    def get_aqa_class(self, vid_feats):
        pred_vid_feats = vid_feats.mean(dim=1)  # (B x 300 x 768)  ->  (B x 768)
        B = pred_vid_feats.shape[0]

        class_pred = F.relu(self.regressor_aqa_1_class(pred_vid_feats))
        class_pred = F.relu(self.regressor_aqa_2_class(class_pred))
        class_pred = self.regressor_aqa_3_class(class_pred)

        return class_pred

    def forward(self, batch, is_decode=False):
        batch = defaultdict(lambda: None, batch)

        if is_decode:
            return self.generate(batch)
        else:
            return self.encode_forward(batch)

    def encode_forward(self, batch):
        if batch["input_ids"] is None:
            img, txt, mask = batch["img"], batch["txt"], batch["mask"]
            ans_mtm = batch["ans_mtm"]
            prompt = batch["prompt"]
            gt_score = batch["gt_score"]
            gt_class = batch["gt_class"]
            sport_names = []
            for i in range(gt_class.shape[0]):
                # 取出当前样本的 class ID (这是一个单元素 Tensor)
                class_id_tensor_i = gt_class[i]
                # 将单元素 Tensor 转换为 Python 整数
                class_id_scalar_i = int(class_id_tensor_i.item())
                # 使用 Python 整数作为索引查找对应的 sport_name
                sport_name_i = self.items_list[class_id_scalar_i][0].lower()
                # 将找到的 sport_name 添加到列表中
                sport_names.append(sport_name_i)  # 需要改

            (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
            _h, _w = _H // 32, _W // 32
            feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
            # Collect general instruction prompts for feat_img1
            batched_general_prompts = []
            for i in range(_B):
                class_id_scalar_i = int(gt_class[i].item())
                sport_name_i = self.items_list[class_id_scalar_i][0].lower()
                sports_steps = " ".join(self.items_list[class_id_scalar_i][1])
                split_parts = re.split(r'(action step \d+: |general instruction: )', sports_steps)
                steps_list = []
                for j in range(1, len(split_parts), 2):
                    step_header = split_parts[j]
                    step_content = split_parts[j+1].strip()
                    steps_list.append(step_header + step_content)
                general_steps_only = [s for s in steps_list if s.startswith("general instruction")]
                if general_steps_only:
                    general_text = general_steps_only[0]
                    if len(general_text) > 256:
                        processed_general = general_text[:256]
                    else:
                        processed_general = general_text.ljust(256, " ")
                else:
                    processed_general = "".ljust(256, " ")
                batched_general_prompts.append(processed_general)

            if self.use_prompt:
                vid_feats = feat_img
                prompt_emb_class = self.clip_text_model_class(batched_general_prompts)
                # 使用.mean(1)方法对第二个维度进行平均，以得到每个prompt的平均embedding。这样处理后，prompt embedding的形状变为了(score_emb_num x 512)
                
                prompt_emb_class = (
                    prompt_emb_class.mean(1).unsqueeze(0).repeat(_B, 1, 1)
                )  # (B x score_emb_num x 512)

                # Cross Attention-1. K is the video features, Q is the prompt features from CLIPtext encoder. [Modified by Sule]
                # refine the prompt embedding
                prompt_emb_cross, attention_probs = self.cross_att_1(
                    prompt_emb_class, vid_feats
                )
                prompt_emb_class = (
                    prompt_emb_class + self.weight_for_prompt * prompt_emb_cross
                )

                vid_feats_cross, attention_probs = self.cross_att_2(
                    vid_feats, prompt_emb_class
                )  # attention_probs: (B x num_heads x 784 x score_emb_num)
                
                vid_feats = (
                    vid_feats + self.weight_for_vidfeats * vid_feats_cross
                )  # (B x 784 x 512) --> (B x 784 x 512)
                
                # Prepare VL transformer inputs
                feat_img1 = vid_feats
            
            if (
                self.use_prompt
            ):  # 实际和上面的self.use_prompt不一样,实际为self.args.enable_prompt
                vid_feats = feat_img
                num_steps_to_process = 5
                batched_steps_prompts = [[] for _ in range(num_steps_to_process)]
                pattern = r'(action step \d+: |general instruction: )'

                for i in range(_B):  # _B 是 batch_size

                    if gt_score[i].item() > 0.5:
                        score_word = "bad"
                    else:
                        score_word = "good"
                    class_id_scalar_i = int(gt_class[i].item())
                    sport_name_i = self.items_list[class_id_scalar_i][0].lower()
                    sports_steps = " ".join(self.items_list[class_id_scalar_i][1])


                    split_parts = re.split(pattern, sports_steps)
                    steps_list = []
                    for j in range(1, len(split_parts), 2):
                        step_header = split_parts[j]
                        step_content = split_parts[j+1].strip()
                        steps_list.append(step_header + step_content)

                    action_steps_only = [s for s in steps_list if s.startswith("action step")]

                    for step_idx in range(num_steps_to_process):
                        if step_idx < len(action_steps_only):

                            step_text = action_steps_only[step_idx]
                            if len(step_text) > 256:
                                processed_text = step_text[:256]
                            else:
                                processed_text = step_text.ljust(256, " ")
                        else:

                            processed_text = "".ljust(256, " ")
                        batched_steps_prompts[step_idx].append(processed_text)

                for step_idx in range(num_steps_to_process):
                    current_batch_for_step = batched_steps_prompts[step_idx]
                    prompt_emb_class = self.clip_text_model_step(current_batch_for_step)
                    prompt_emb_class = (
                        prompt_emb_class.mean(1).unsqueeze(0).repeat(_B, 1, 1)
                    )  # (B x score_emb_num x 512)
                    prompt_emb_cross, attention_probs = self.cross_att_5(
                        prompt_emb_class, vid_feats
                    )
                    
                    #     prompt_emb_class = (
                    #     prompt_emb_class + self.weight_for_prompt * prompt_emb_cross
                    # )
                    vid_feats_cross, attention_probs = self.cross_att_6(
                        vid_feats, prompt_emb_class
                    )  # attention_probs: (B x num_heads x 784 x score_emb_num)
                    # vid_feats = (
                    #     vid_feats + self.weight_for_vidfeats * vid_feats_cross
                    # )  # (B x 784 x 512) --> (B x 784 x 512)
                    gate = T.sigmoid(self.gate_projection(T.cat([vid_feats, vid_feats_cross], dim=-1)))
                    vid_feats = (1 - gate) * vid_feats + gate * vid_feats_cross
                    # Prepare VL transformer inputs
                    # prompt_summary = prompt_emb_cross.mean(dim=1, keepdim=True)
                    # N_video = vid_feats_cross.shape[1]
                    # expanded_prompt = prompt_summary.expand(-1, N_video, -1)
                    # combined_features = T.cat([vid_feats_cross, expanded_prompt], dim=2)
                    # fused_features = self.fusion_projection(combined_features)
                    # vid_feats = fused_features
                    feat_img2 = vid_feats
            
                
                concatenated_feat = T.cat([feat_img1, feat_img2], dim=-1) 
                gate = self.fusion_gate_net(concatenated_feat)
                feat_img = gate * feat_img1 + (1 - gate) * feat_img2
            pred_score = self.get_aqa_score(feat_img)
            pred_score = pred_score.squeeze(1)
            pred_score_last = T.sigmoid(pred_score)
            pred_class = self.get_aqa_class(feat_img)
            # pred_class = pred_class.squeeze(1)
            batch["pred_score"] = pred_score
            batch["pred_class"] = pred_class
            ans_mtm, _, feat_txt = self.prepro_txt_inputs(
                ans_mtm, mask_txt, feat_txt, task_name="cap", prompt=prompt
            )
            if prompt is not None and self.args.enable_prompt:
                _L = len(prompt[0])
                self.cap_prompt_txt_L = _L
            elif self.args.enable_task_token:
                _L = 1  # for a task token
                self.cap_prompt_txt_L = _L
            else:
                _L = 0
                assert self.cap_prompt_txt_L == _L
            ans_mtm[:, :_L] = -1
            if _L > 0:
                mask_pretxt = T.ones_like(mask_txt)[:, :_L]
            else:
                mask_pretxt = None

            out, _ = self.go_cross(
                feat_img,
                mask_img,
                feat_txt,
                mask_txt,
                attn_mask_type=batch["attn_mask_type"],
                mask_pretxt=mask_pretxt,
            )
            out = self.fc_mtm(out[:, (1 + _h * _w) * _T :])
            return {
                "out": out,
                "ans": ans_mtm,
                "pred_score": pred_score,
                "pred_class": pred_class,
                "gt_score": gt_score,
                "gt_class": gt_class,
            }
        else:
            input_ids = batch["input_ids"]
            feat_txt = self.enc_txt(input_ids, attn_mask_type="seq2seq")
            txt_seq_len = input_ids.shape[-1]
            feat_img = batch["feat_img"]
            mask = batch["attention_mask"]
            cap_pretxt_feat = batch["cap_pretxt_feat"]
            # prompt = batch["prompt"]
            if feat_img is not None:
                _, img_seq_len, _ = feat_img.shape
                if cap_pretxt_feat is not None:
                    feat = T.cat([feat_img, cap_pretxt_feat, feat_txt], dim=1)
                    out_seq_start = img_seq_len + self.cap_prompt_txt_L
                    out_seq_end = img_seq_len + self.cap_prompt_txt_L + txt_seq_len
                else:
                    feat = T.cat([feat_img, feat_txt], dim=1)
                    out_seq_start = img_seq_len
                    out_seq_end = img_seq_len + txt_seq_len
                mask = self.mask_ext(mask, mask.shape, mask.device)
                # safeguard fp16
                mask = mask.to(dtype=feat_txt.dtype)
                outputs = self.trsfr(feat, mask, output_attentions=True, use_cache=True)
                sequence_output = outputs["last_hidden_state"][
                    :, out_seq_start:out_seq_end, :
                ]
            else:
                raise NotImplementedError(
                    "Fast decoding with past key-values is not validated"
                )
                assert not self.args.enable_task_token
                assert not self.args.enable_prompt
                feat = feat_txt
                past = batch["past_key_values"]
                mask = self.mask_ext(mask, mask.shape, mask.device)
                outputs = self.trsfr(
                    feat,
                    mask,
                    output_attentions=True,
                    use_cache=True,
                    past_key_values=past,
                )
                sequence_output = outputs["last_hidden_state"][:, :txt_seq_len, :]
            class_logits = self.fc_mtm(sequence_output)
            past_key_values = outputs["past_key_values"]
            return {"logits": class_logits, "past": past_key_values}

    def test(self, batch):
        batch = defaultdict(lambda: None, batch)
        img, txt, mask = batch["img"], batch["txt"], batch["mask"]
        ans_mtm = batch["ans_mtm"]
        prompt = batch["prompt"]
        gt_score = batch["gt_score"]
        gt_class = batch["gt_class"]
        sport_names = []
        for i in range(gt_class.shape[0]):
            # 取出当前样本的 class ID (这是一个单元素 Tensor)
            class_id_tensor_i = gt_class[i]
            # 将单元素 Tensor 转换为 Python 整数
            class_id_scalar_i = int(class_id_tensor_i.item())
            # 使用 Python 整数作为索引查找对应的 sport_name
            sport_name_i = self.items_list[class_id_scalar_i][0].lower()
            # 将找到的 sport_name 添加到列表中
            sport_names.append(sport_name_i)  # 需要改

        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H // 32, _W // 32
        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
        if self.use_prompt:
            vid_feats = feat_img
            prompt_emb_score = self.clip_text_model_score(0)
            prompt_emb_class = self.clip_text_model_class(
                0
            )  # (score_emb_num x 14 x 512) prompt embedding from text encoder

            # 使用.mean(1)方法对第二个维度进行平均，以得到每个prompt的平均embedding。这样处理后，prompt embedding的形状变为了(score_emb_num x 512)
            prompt_emb_score = prompt_emb_score.mean(1).unsqueeze(0).repeat(_B, 1, 1)
            prompt_emb_class = (
                prompt_emb_class.mean(1).unsqueeze(0).repeat(_B, 1, 1)
            )  # (B x score_emb_num x 512)

            # Cross Attention-1. K is the video features, Q is the prompt features from CLIPtext encoder. [Modified by Sule]
            # refine the prompt embedding
            prompt_emb_cross, attention_probs = self.cross_att_1(
                prompt_emb_class, vid_feats
            )
            prompt_emb_class = (
                prompt_emb_class + self.weight_for_prompt * prompt_emb_cross
            )

            prompt_emb_score_cross, attention_probs = self.cross_att_3(
                prompt_emb_score, vid_feats
            )
            prompt_emb_score = (
                prompt_emb_score + self.weight_for_prompt * prompt_emb_score_cross
            )
            # Cross Attention-2. Q is the video features, K is the prompt features from CLIPtext encoder. [Modified by Shiyi Zhang]
            # refine the visual feats

            vid_feats_cross, attention_probs = self.cross_att_2(
                vid_feats, prompt_emb_class
            )  # attention_probs: (B x num_heads x 784 x score_emb_num)
            vid_feats_cross, attention_probs = self.cross_att_4(
                vid_feats, prompt_emb_score
            )
            vid_feats = (
                vid_feats + self.weight_for_vidfeats * vid_feats_cross
            )  # (B x 784 x 512) --> (B x 784 x 512)

            # Prepare VL transformer inputs
            feat_img = vid_feats
        pred_score = self.get_aqa_score(feat_img)
        pred_score = pred_score.squeeze(1)
        pred_score_last = T.sigmoid(pred_score)
        pred_class = self.get_aqa_class(feat_img)
        # pred_class = pred_class.squeeze(1)
        batch["pred_score"] = pred_score
        batch["pred_class"] = pred_class
        return {
            "pred_score": pred_score,
            "pred_class": pred_class,
            "gt_score": gt_score,
            "gt_class": gt_class,
        }

    def generate(self, batch):
        """Generates captions given image features"""
        in_img, in_txt, in_mask = batch["img"], batch["txt"], batch["mask"]
        (_B, _, _, _, _) = in_img.shape
        bos_token_id = self.cls_token_id
        pad_token_id = self.pad_token_id
        eos_token_ids = [self.sep_token_id]

        # default settings
        # # NOTE: num_keep_best is not equavilant to num_return_sequences
        # # num_keep_best is the number of hypotheses to keep in beam search
        # # num_return_sequences is the repeating times of input, coupled with
        # # do_sample=True can generate more than one samples per image
        self.num_keep_best = self.args.get("num_keep_best", 1)
        num_beams = self.args.get("num_beams", 1)
        num_return_sequences = self.args.get("num_return_sequences", 1)
        num_fsm_states = self.args.get("num_fsm_states", 1)
        do_sample = self.args.get("do_sample", False)
        temperature = self.args.get("gen_temperature", 1.0)
        top_k = self.args.get("top_k", 0)
        top_p = self.args.get("top_p", 1)
        repetition_penalty = self.args.get("repetition_penalty", 1)

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(in_img, in_txt, in_mask)
        prompt = batch["prompt"]
        gt_score = batch["gt_score"]
        gt_class = batch["gt_class"]
        # Collect general instruction prompts for feat_img1
        batched_general_prompts = []
        for i in range(_B):
            class_id_scalar_i = int(gt_class[i].item())
            sport_name_i = self.items_list[class_id_scalar_i][0].lower()
            sports_steps = " ".join(self.items_list[class_id_scalar_i][1])
            split_parts = re.split(r'(action step \d+: |general instruction: )', sports_steps)
            steps_list = []
            for j in range(1, len(split_parts), 2):
                step_header = split_parts[j]
                step_content = split_parts[j+1].strip()
                steps_list.append(step_header + step_content)
            general_steps_only = [s for s in steps_list if s.startswith("general instruction")]
            if general_steps_only:
                general_text = general_steps_only[0]
                if len(general_text) > 256:
                    processed_general = general_text[:256]
                else:
                    processed_general = general_text.ljust(256, " ")
            else:
                processed_general = "".ljust(256, " ")
            batched_general_prompts.append(processed_general)

        if self.use_prompt:
            vid_feats = feat_img
            prompt_emb_class = self.clip_text_model_class(batched_general_prompts)
            prompt_emb_class = (
                prompt_emb_class.mean(1).unsqueeze(0).repeat(_B, 1, 1)
            )  # (B x score_emb_num x 512)
            prompt_emb_cross, attention_probs = self.cross_att_1(
                prompt_emb_class, vid_feats
            )
            prompt_emb_class = (
                prompt_emb_class + self.weight_for_prompt * prompt_emb_cross
            )

            vid_feats_cross, attention_probs = self.cross_att_2(
                vid_feats, prompt_emb_class
            )  # attention_probs: (B x num_heads x 784 x score_emb_num)
            vid_feats = (
                vid_feats + self.weight_for_vidfeats * vid_feats_cross
            )  # (B x 784 x 512) --> (B x 784 x 512)

            # Prepare VL transformer inputs
            feat_img1 = vid_feats
        
        # pred_class = pred_class.squeeze(1)

        if (
                self.use_prompt
            ):  # 实际和上面的self.use_prompt不一样,实际为self.args.enable_prompt
                vid_feats = feat_img
                num_steps_to_process = 5
                batched_steps_prompts = [[] for _ in range(num_steps_to_process)]
                pattern = r'(action step \d+: |general instruction: )'

                for i in range(_B):  # _B 是 batch_size

                    if gt_score[i].item() > 0.5:
                        score_word = "bad"
                    else:
                        score_word = "good"
                    class_id_scalar_i = int(gt_class[i].item())
                    sport_name_i = self.items_list[class_id_scalar_i][0].lower()
                    sports_steps = " ".join(self.items_list[class_id_scalar_i][1])


                    split_parts = re.split(pattern, sports_steps)
                    steps_list = []
                    for j in range(1, len(split_parts), 2): 
                        step_header = split_parts[j]
                        step_content = split_parts[j+1].strip()
                        steps_list.append(step_header + step_content)
                    
                    action_steps_only = [s for s in steps_list if s.startswith("action step")]
                    
                
                    for step_idx in range(num_steps_to_process):
                        if step_idx < len(action_steps_only):

                            step_text = action_steps_only[step_idx]
                            if len(step_text) > 256:
                                processed_text = step_text[:256]
                            else:
                                processed_text = step_text.ljust(256, " ")
                        else:

                            processed_text = "".ljust(256, " ")
                        batched_steps_prompts[step_idx].append(processed_text)

                for step_idx in range(num_steps_to_process):
                    current_batch_for_step = batched_steps_prompts[step_idx]
                    prompt_emb_class = self.clip_text_model_step(current_batch_for_step)
                    prompt_emb_class = (
                        prompt_emb_class.mean(1).unsqueeze(0).repeat(_B, 1, 1)
                    )  # (B x score_emb_num x 512)
                    prompt_emb_cross, attention_probs = self.cross_att_5(
                        prompt_emb_class, vid_feats
                    )
                    
                    #     prompt_emb_class = (
                    #     prompt_emb_class + self.weight_for_prompt * prompt_emb_cross
                    # )
                    vid_feats_cross, attention_probs = self.cross_att_6(
                        vid_feats, prompt_emb_class
                    )  # attention_probs: (B x num_heads x 784 x score_emb_num)
                    # vid_feats = (
                    #     vid_feats + self.weight_for_vidfeats * vid_feats_cross
                    # )  # (B x 784 x 512) --> (B x 784 x 512)
                    gate = T.sigmoid(self.gate_projection(T.cat([vid_feats, vid_feats_cross], dim=-1)))
                    vid_feats = (1 - gate) * vid_feats + gate * vid_feats_cross
                    # Prepare VL transformer inputs
                    # prompt_summary = prompt_emb_cross.mean(dim=1, keepdim=True)
                    # N_video = vid_feats_cross.shape[1]
                    # expanded_prompt = prompt_summary.expand(-1, N_video, -1)
                    # combined_features = T.cat([vid_feats_cross, expanded_prompt], dim=2)
                    # fused_features = self.fusion_projection(combined_features)
                    # vid_feats = fused_features
                    feat_img2 = vid_feats
                concatenated_feat = T.cat([feat_img1, feat_img2], dim=-1) 
                gate = self.fusion_gate_net(concatenated_feat)
                feat_img = gate * feat_img1 + (1 - gate) * feat_img2
        pred_score = self.get_aqa_score(feat_img)
        pred_score = pred_score.squeeze(1)
        pred_score_last = T.sigmoid(pred_score)
        pred_class = self.get_aqa_class(feat_img)
        _, cap_pretxt_mask, cap_pretxt_feat = self.get_pretxt(
            mask_txt, task_name="cap", prompt=prompt
        )
        if cap_pretxt_mask is not None:
            self.cap_prompt_txt_L = cap_pretxt_mask.shape[1]
        else:
            # _L = 0  # no added task token or prompt
            assert self.cap_prompt_txt_L == 0
        self.cap_pretxt_feat = cap_pretxt_feat
        attention_mask = self.get_attn_mask(
            mask_img,
            mask_txt,
            attn_mask_type=batch["attn_mask_type"],
            mask_pretxt=cap_pretxt_mask,
        )
        max_length = self.args.get("max_gen_length", 20)
        batch_size = feat_img.shape[0]
        self.img_seq_len = feat_img.shape[1]
        self.max_seq_len = feat_txt.shape[1]
        # max(self.args.size_txt, feat_txt.shape[1])
        self.past_key_values = None

        assert feat_txt.shape == (batch_size, self.max_seq_len, self.hidden_size)
        input_ids = None

        if input_ids is None:
            input_ids = T.full(
                (batch_size, 1),
                bos_token_id,
                dtype=T.long,
                device=next(self.parameters()).device,
            )

            if self.use_texthead:
                custom_prompt_strings = []
                for i in range(_B):
                    if pred_score_last[i].item() > 0.5:
                        score_word = "bad"
                    else:
                        score_word = "good"
                    # 取出当前样本的 class ID (这是一个单元素 Tensor)
                    class_id_tensor_i = gt_class[i]
                    # 将单元素 Tensor 转换为 Python 整数
                    class_id_scalar_i = int(class_id_tensor_i.item())
                    # 使用 Python 整数作为索引查找对应的 sport_name
                    sport_name_i = self.items_list[class_id_scalar_i][0].lower()
                    prompt_str = (
                        f"The {sport_name_i} action in this video is {score_word}."
                    )
                    custom_prompt_strings.append(prompt_str)
                prompt_tokens = self.tokzr(
                    custom_prompt_strings,
                    padding=True,  # Pad prompts to the max length within the batch
                    return_tensors="pt",
                ).to(feat_img.device)  # type: ignore
                prompt_token_ids = prompt_tokens["input_ids"]  # (B, prompt_len)
                cleaned_prompt_ids = []
                prompt_lengths = []
                for ids in prompt_token_ids:
                    # Filter out CLS, SEP, and PAD
                    cleaned_ids = [
                        id
                        for id in ids.tolist()
                        if id not in [bos_token_id, pad_token_id, eos_token_ids]
                    ]
                    # Add SEP back at the end if desired, or just use the core sentence tokens
                    # cleaned_ids.append(sep_id) # Optional: add SEP to mark end of prompt sentence
                    cleaned_prompt_ids.append(cleaned_ids)
                    prompt_lengths.append(len(cleaned_ids))
                # Pad the cleaned prompt IDs again if lengths are different after removing specials
                max_prompt_len = max(prompt_lengths) if prompt_lengths else 0
                padded_cleaned_prompt_ids = []
                for ids in cleaned_prompt_ids:
                    padded_ids = ids + [pad_token_id] * (max_prompt_len - len(ids))
                    padded_cleaned_prompt_ids.append(padded_ids)
                prompt_token_ids = T.tensor(padded_cleaned_prompt_ids, dtype=T.long).to(
                    feat_img.device
                )

                # input_ids = T.cat([input_ids, prompt], dim=1)
                input_ids = T.cat([input_ids, prompt_token_ids], dim=1)
            # input_ids = T.cat(input_ids, input_ids], dim=1)
        else:
            assert input_ids.dim() == 2, (
                f"Input prompt of shape {input_ids.shape()}"
                + "should be of shape (batch_size, sequence length)."
            )
            assert input_ids.shape[0] == batch_size, (
                f"Input prompt of shape {input_ids.shape()}"
                + f"Input batch size must match image batch size {batch_size}"
            )

        cur_len = input_ids.shape[1]
        assert num_return_sequences == 1, (
            "Only supporting num_return_sequences == 1, but got "
            + f"{num_return_sequences} instead"
        )
        effective_batch_size = batch_size

        num_expand = num_beams * num_fsm_states * num_return_sequences
        self.img_feats = self._expand_for_beams(feat_img, num_expand)
        if self.cap_pretxt_feat is not None:
            self.cap_pretxt_feat = self._expand_for_beams(
                self.cap_pretxt_feat, num_expand
            )
        self.full_attention_mask = self._expand_for_beams(attention_mask, num_expand)

        output = self._generate_no_beam_search(
            input_ids,
            cur_len,
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            pad_token_id,
            eos_token_ids,
            effective_batch_size,
        )
        # if self.cap_prompt_txt_L > 0:
        #     output = (
        #         output[0][:, :, self.cap_prompt_txt_L:], output[1])
        batch["pred_score"] = pred_score
        batch["pred_class"] = pred_class
        return output

    def _expand_for_beams(self, x, num_expand):
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_expand, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x

    def prepare_inputs_for_generation(self, curr_ids, past=None):
        # NOTE: if attention is on,
        # it should be the token used to mask words in training
        mask_token_id = self.mask_token_id
        batch_size = curr_ids.shape[0]
        mask_ids = T.full(
            (batch_size, 1), mask_token_id, dtype=T.long, device=curr_ids.device
        )

        def _slice(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len)
            return t[:, start:end]

        def _remove_elements(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len)
            return T.cat([t[:, :start], t[:, end:]], dim=1)

        if past is None:
            input_ids = T.cat([curr_ids, mask_ids], dim=1)

            curr_len = input_ids.shape[1]
            cap_pretxt_feat = self.cap_pretxt_feat
            if cap_pretxt_feat is None:
                assert self.cap_prompt_txt_L == 0
            full_len = self.max_seq_len + self.cap_prompt_txt_L + self.img_seq_len
            assert self.full_attention_mask.shape == (batch_size, full_len, full_len)

            def _remove_rows_cols(t, row_start, row_end, col_start, col_end):
                t00 = t[:, :row_start, :col_start]
                t01 = t[:, :row_start, col_end:]
                t10 = t[:, row_end:, :col_start]
                t11 = t[:, row_end:, col_end:]
                res = T.cat([T.cat([t00, t01], dim=2), T.cat([t10, t11], dim=2)], dim=1)
                assert res.shape == (
                    t.shape[0],
                    t.shape[1] - row_end + row_start,
                    t.shape[2] - col_end + col_start,
                )
                return res

            img_feats = self.img_feats
            seq_end = self.img_seq_len + self.cap_prompt_txt_L + curr_len

            # seq_start = curr_len
            # seq_end = self.max_seq_len
            # attention_mask = _remove_rows_cols(
            #     self.full_attention_mask, seq_start,
            #     seq_end, seq_start, seq_end)
            attention_mask = self.full_attention_mask[:, :seq_end, :seq_end]
            past_key_values = None
        else:
            raise NotImplementedError(
                "Fast decoding with past key-value is not validated"
            )
            last_token = curr_ids[:, -1:]
            # The representation of last token should be re-computed, because
            # it depends on both self-attention context and input tensor
            input_ids = T.cat([last_token, mask_ids], dim=1)
            start_pos = curr_ids.shape[1] - 1
            end_pos = start_pos + input_ids.shape[1]

            img_feats = None
            assert past[0][0].shape[2] == 1 + self.img_seq_len + start_pos
            past_key_values = []
            for layer_idx in range(len(past)):
                key, value = past[layer_idx]
                past_key_values.append(
                    (
                        key[:, :, : self.img_seq_len + start_pos],
                        value[:, :, : self.img_seq_len + start_pos],
                    )
                )

            # assert past[0].shape[0] == batch_size
            # past is a list of key values
            # key value of shape [
            #   batch_size, number_heads, seq_len, attn_hidden_size]
            # if self.past_key_values is None:
            #     assert start_pos == 1  # the first token after BOS
            #     assert past[0][0].shape[2] == 2 + self.img_seq_len
            #     self.past_key_values = []
            #     for layer_idx in range(len(past)):
            #         key, value = past[layer_idx]
            #         self.past_key_values.append((
            #             key[:, :, :self.img_seq_len + start_pos],
            #             value[:, :, :self.img_seq_len + start_pos],
            #             ))
            #     # # reorder to [img_feats, sentence]
            #     # self.prev_encoded_layers = [
            #     #         T.cat([x[:, 2:, :], x[:, :start_pos, :]], dim=1)
            #     #         for x in past]
            #     # i2i = self.full_attention_mask[
            #     #     :, :self.img_seq_len, :self.img_seq_len]
            #     # i2s = self.full_attention_mask[
            #     #     :, :self.img_seq_len, self.img_seq_len:]
            #     # s2i = self.full_attention_mask[
            #     #     :, self.img_seq_len:, :self.img_seq_len]
            #     # s2s = self.full_attention_mask[
            #     #     :, self.img_seq_len:, self.img_seq_len:]
            #     # self.full_attention_mask = T.cat(
            #     #         [T.cat([i2i, i2s], dim=2),
            #     #          T.cat([s2i, s2s], dim=2)], dim=1)
            # else:
            #     assert start_pos > 1
            #     assert past[0][0].shape[2] == 2
            #     # self.prev_encoded_layers = [
            #     #     T.cat([x, p[:, :-1, :]], dim=1)
            #     #     for x, p in zip(self.prev_encoded_layers, past)]
            #     for layer_idx in range(len(past)):
            #         key, value = past[layer_idx]
            #         p_key, p_value = self.past_key_values[layer_idx]
            #         new_key = T.cat([p_key, key[:, :, :-1, :]], dim=2)
            #         new_value = T.cat([p_value, value[:, :, :-1, :]], dim=2)
            #         self.past_key_values[layer_idx] = (new_key, new_value)

            attention_mask = self.full_attention_mask[
                :,
                self.img_seq_len + start_pos : self.img_seq_len + end_pos,
                : self.img_seq_len + end_pos,
            ]

        return {
            "input_ids": input_ids,
            "feat_img": img_feats,
            "cap_pretxt_feat": cap_pretxt_feat,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "task": "captioning_generation",
        }

    def _do_output_past(self):
        return self.config.is_decoder

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        assert self.num_keep_best == 1, "cannot generate >1 sentences in greedy search"
        # current position / max lengths /
        # length of generated sentences / unfinished sentences
        unfinished_sents = []
        if T._C._get_tracing_state():
            cur_unfinished = T.ones(1, dtype=input_ids)
        else:
            cur_unfinished = input_ids.new(batch_size).fill_(1)

        # log of scores for each sentence in the batch
        logprobs = []

        past = None

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(model_inputs)
            logits = outputs["logits"]

            if cur_len == 1:
                token_len = 2
                next_token_idx = 1
            else:
                assert cur_len > 1
                if not self._do_output_past():
                    token_len = cur_len + 1
                    next_token_idx = cur_len
                else:
                    token_len = 2
                    next_token_idx = 1

            assert logits.shape[1] == token_len
            next_token_logits = logits[:, next_token_idx, :]

            # if model has past, then set
            # the past variable to speed up decoding
            if self._do_output_past():
                past = outputs["past"]

            # repetition penalty from CTRL paper
            # (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to
                        # multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature =>
                #   more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(
                    next_token_logits, top_k=top_k, top_p=top_p
                )
                # Sample
                next_token = T.multinomial(
                    F.softmax(next_token_logits, dim=-1), num_samples=1
                ).squeeze(1)
            else:
                # Greedy decoding
                next_token = T.argmax(next_token_logits, dim=-1)

            # Compute scores
            _scores = F.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size, vocab_size)
            _scores = T.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)

            # update generations and finished sentences
            tokens_to_add = next_token * cur_unfinished + pad_token_id * (
                1 - cur_unfinished
            )
            input_ids = T.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            # for t in input_ids:
            #     print(self.tokenizer.convert_ids_to_tokens(t.tolist()))

            for eos_token_id in eos_token_ids:
                cur_unfinished = cur_unfinished.mul(
                    tokens_to_add.ne(eos_token_id).long()
                )
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence,
            # or if we exceed the maximul length
            if cur_unfinished.max() == 0:
                break

        # add eos_token_ids to unfinished sentences
        if cur_len == max_length:
            input_ids[:, -1].masked_fill_(
                cur_unfinished.to(dtype=T.bool), eos_token_ids[0]
            )

        logprobs = T.cat(logprobs, dim=1)
        unfinished_sents = T.stack(unfinished_sents, dim=1).float()
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
        # return logprobs to keep consistent with beam search output
        logprobs = sum_logprobs / unfinished_sents.sum(dim=1)

        # pad to the same length, otherwise DataParallel will give error
        pad_len = max_length - input_ids.shape[1]
        if pad_len > 0:
            padding_ids = input_ids.new(batch_size, pad_len).fill_(pad_token_id)
            input_ids = T.cat([input_ids, padding_ids], dim=1)

        # (batch_size, n_best, max_len), (batch_size, n_best)
        return input_ids.unsqueeze(1), logprobs.unsqueeze(1)


def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < T.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = T.sort(logits, descending=True)
        cumulative_probs = T.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits
