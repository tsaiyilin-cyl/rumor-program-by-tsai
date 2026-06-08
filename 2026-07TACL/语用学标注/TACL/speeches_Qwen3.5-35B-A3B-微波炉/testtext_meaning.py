# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer, util

# 1. 初始化模型（使用 all-mpnet-base-v2）
model = SentenceTransformer('all-mpnet-base-v2')
new_sentence = "我坚信美国登月是真的"

# 2. 定义多句基线（每个立场多条表述）
base_sentences = {
    "支持": [
        "我认为美国成功登月不是阴谋论",
        "我坚信美国登月是真实的",
        "美国登月毫无疑问是事实"
    ],
    "中立": [
        "对于美国成功登月是否是阴谋论，我持中立观点",
        "我不确定登月是否真实",
        "这个问题尚无定论"
    ],
    "反对": [
        "我认为美国成功登月是阴谋论",
        "我怀疑美国登月的真实性",
        "登月很可能是伪造的"
    ]
}

# 3. 将字典展平为句子列表 + 对应标签列表（方便后续打印）
all_sentences = []
all_labels = []
for label, sentences in base_sentences.items():
    for sent in sentences:
        all_sentences.append(sent)
        all_labels.append(label)

# 4. 编码所有句子（包括新句子）
print("正在编码句子...")
all_embeddings = model.encode(all_sentences, convert_to_tensor=True)
new_emb = model.encode(new_sentence, convert_to_tensor=True)

# 5. 计算新句子与每个基线句子的相似度
print("\n新句子与每个基线句子的相似度：")
sims_with_each = util.cos_sim(new_emb, all_embeddings)[0]  # 得到一个长度为句子总数的向量
for i, (sent, label) in enumerate(zip(all_sentences, all_labels)):
    print(f"  [{label}] {sent}")
    print(f"    相似度: {sims_with_each[i].item():.4f}")

# 6. 计算每个类别的平均向量
class_embeddings = {}
for label, sentences in base_sentences.items():
    emb = model.encode(sentences, convert_to_tensor=True)
    class_embeddings[label] = emb.mean(dim=0)  # 平均向量

# 7. 计算与各类别平均向量的相似度
similarities = {label: util.cos_sim(new_emb, class_emb)[0].item()
                for label, class_emb in class_embeddings.items()}

print("\n新句子与各类别的平均相似度：")
for label, sim in similarities.items():
    print(f"  {label}: {sim:.4f}")

# 8. 找出最高相似度的类别
max_label = max(similarities, key=similarities.get)
max_sim = similarities[max_label]

print(f"\n 分类结果：{max_label} (相似度: {max_sim:.4f})")