import json
from collections import defaultdict
from pycocoevalcap.caption_eval import COCOEvalCap
# COCOEvalCap 依赖于 COCO 数据结构，为了简化，我们可以模拟一个简单的 COCO 结构
# 或者使用更直接的 scorer，但 COCOEvalCap 更标准，我们模拟它。
# 或者使用 pycocoevalcap.eval_tabs.CaptionScorer 更简单，我们提供两种方式。


# 方式一：使用更底层的 COCOEvalCap (更接近 COCO 标准评估)
class COCO_Simulated:
    def __init__(self, data):
        # data is a dictionary {image_id: [{'caption': '...', 'id': ..., 'image_id': ...}]}
        self.anns = {item["id"]: item for items in data.values() for item in items}
        self.imgToAnns = data
        self.imgs = {
            img_id: [{"id": img_id}] for img_id in data.keys()
        }  # Just minimal info


def evaluate_with_cocoevalcap(ground_truth_path, generated_path):
    """使用 pycocoevalcap 评估 BLEU, CIDEr 等指标"""

    gts = defaultdict(
        list
    )  # ground truth captions {image_id: [caption1, caption2, ...]}
    res = {}  # generated captions {image_id: [caption]} - pycocoevalcap expects a list even for one caption

    # 加载真实文本
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_id = parts[0]
            caption = parts[1]
            # pycocoevalcap needs a specific structure for ground truth
            gts[img_id].append(
                {"image_id": img_id, "id": len(gts[img_id]), "caption": caption}
            )

    # 加载预测文本
    with open(generated_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_id = parts[0]
            try:
                gen_list = json.loads(parts[1])
                if gen_list and "caption" in gen_list[0]:
                    # pycocoevalcap needs a specific structure for results
                    # Assuming the first caption in the list is the main prediction
                    res[img_id] = [
                        {"image_id": img_id, "caption": gen_list[0]["caption"]}
                    ]
            except json.JSONDecodeError:
                print(f"Skipping line due to JSON error: {line.strip()}")
                continue
            except KeyError:
                print(
                    f"Skipping line due to missing 'caption' key in JSON: {line.strip()}"
                )
                continue

    # 确保只评估真实和预测都存在的 ID
    common_ids = set(gts.keys()) & set(res.keys())
    gts_filtered = {img_id: gts[img_id] for img_id in common_ids}
    res_filtered = {img_id: res[img_id] for img_id in common_ids}

    # 使用模拟的 COCO 数据结构
    coco_gt = COCO_Simulated(gts_filtered)
    coco_res = COCO_Simulated(
        res_filtered
    )  # For res, each img_id has only one item in its list

    # Run evaluation
    cocoEval = COCOEvalCap(coco_gt, coco_res)
    # cocoEval.params['image_id'] = coco_res.getImgIds() # Should use common_ids instead
    cocoEval.params["image_id"] = list(common_ids)

    cocoEval.evaluate()

    # Print results
    print("\n--- pycocoevalcap Results ---")
    for metric, score in cocoEval.eval.items():
        print(f"{metric}: {score:.3f}")
    print("---------------------------")

    return cocoEval.eval


# 方式二：使用 bert_score (需要处理引用格式)
import torch
from bert_score import score
from tqdm import tqdm  # 可选，用于显示进度条


def evaluate_with_bertscore(ground_truth_path, generated_path, lang="en"):
    """使用 bert_score 评估 P, R, F1"""

    cands = []  # Generated captions [caption1, caption2, ...]
    refs = []  # Reference captions [[ref1_1, ref1_2, ...], [ref2_1, ...], ...]
    gts_dict = defaultdict(list)  # {image_id: [caption1, caption2, ...]}
    res_dict = {}  # {image_id: caption} - bert_score cand only needs the string

    # 加载真实文本
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_id = parts[0]
            caption = parts[1]
            gts_dict[img_id].append(caption)

    # 加载预测文本
    with open(generated_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading generated captions for BERTScore"):
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_id = parts[0]
            try:
                gen_list = json.loads(parts[1])
                if gen_list and "caption" in gen_list[0]:
                    # Assuming the first caption is the prediction
                    res_dict[img_id] = gen_list[0]["caption"]
            except json.JSONDecodeError:
                print(f"Skipping line due to JSON error: {line.strip()}")
                continue
            except KeyError:
                print(
                    f"Skipping line due to missing 'caption' key in JSON: {line.strip()}"
                )
                continue

    # 确保只评估真实和预测都存在的 ID，并按相同顺序排列
    common_ids = sorted(list(set(gts_dict.keys()) & set(res_dict.keys())))

    for img_id in common_ids:
        cands.append(res_dict[img_id])
        refs.append(gts_dict[img_id])  # Append the list of references

    if not cands or not refs:
        print(
            "No common IDs found between ground truth and generated files for BERTScore."
        )
        return {}

    # 计算 BERTScore
    print("\n--- Running BERTScore ---")
    # P, R, F1 are tensors of shape [num_samples]
    P, R, F1 = score(cands, refs, lang=lang, verbose=True)
    print("---------------------------")

    # 报告平均 F1, P, R 分数
    avg_P = P.mean().item()
    avg_R = R.mean().item()
    avg_F1 = F1.mean().item()

    print(f"BERTScore-P: {avg_P:.3f}")
    print(f"BERTScore-R: {avg_R:.3f}")
    print(f"BERTScore-F1: {avg_F1:.3f}")

    return {"BERTScore-P": avg_P, "BERTScore-R": avg_R, "BERTScore-F1": avg_F1}


# 示例用法
if __name__ == "__main__":
    # 创建一些模拟的输入文件（用于测试）
    with open("ground_truth.txt", "w", encoding="utf-8") as f:
        f.write("64_01\tThe person performs a standing oblique crunch.\n")
        f.write("64_01\tThis video shows someone doing oblique crunches standing up.\n")
        f.write("64_02\tA different action is shown here.\n")
        f.write("64_03\tThis is the third example reference.\n")

    with open("generated.txt", "w", encoding="utf-8") as f:
        f.write(
            '64_01\t[{"caption": "the standing oblique crunch action in this video is good. the person maintains a straight back...", "conf": 0.5434824228286743}]\n'
        )
        f.write(
            '64_02\t[{"caption": "another predicted caption about a different action.", "conf": 0.6}]\n'
        )
        f.write(
            '64_04\t[{"caption": "this ID only exists in generated.", "conf": 0.7}]\n'
        )  # 存在于预测但不存在于真实
        f.write("invalid_line\n")  # 模拟一个无效行
        f.write("64_05\tinvalid_json\n")  # 模拟一个JSON错误

    # 定义文件路径
    gt_file = "ground_truth.txt"
    gen_file = "generated.txt"

    # 运行 pycocoevalcap 评估
    cocoeval_results = evaluate_with_cocoevalcap(gt_file, gen_file)
    print("\nFull COCOEvalCap Results:", cocoeval_results)

    # 运行 bert_score 评估
    # 确保语言设置正确，对于英文是 'en'
    bertscore_results = evaluate_with_bertscore(gt_file, gen_file, lang="en")
    print("\nFull BERTScore Results:", bertscore_results)
