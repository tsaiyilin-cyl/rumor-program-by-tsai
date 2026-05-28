# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]
sentences = [
    "我认为美国成功登月不是阴谋论",
    "我认为美国成功登月是阴谋论",
    "对于美国成功登月是否是阴谋论，我持中立观点",
    "作为一名普通居民，我认同美国成功登月是真实的历史成就。这一结论基于多方面的公开证据：包括各国科研机构对月球样本的分析、独立追踪到的月球轨道器数据，以及近年来多国探测器拍摄到的阿波罗任务遗迹照片。此外，数十万科技工作者参与的工程难以被长期掩盖，而冷战对手苏联的监测也从未提出质疑。这些事实共同构成了坚实的科学和历史依据。\n\n当然，任何重大事件都值得理性探讨，我也理解人们对权威叙事保持审慎的态度。不知道您对此有什么样的看法？很乐意倾听您的观点。",
    "作为一名普通居民，我确实对“美国成功登月”的历史叙述存有一些疑虑。尽管官方提供了大量影像和科学数据，但考虑到当时的技术条件与政治竞争背景，部分细节——如旗帜在真空中飘动、照片光影不一致等——仍让我觉得存在未完全解释的疑点。科技发展的过程往往是复杂的，而历史叙述有时也可能受到时代语境的影响。我尊重那些坚信登月真实性的人，但也认为公众保持理性追问对科学精神的深化是有益的。\n\n不知道您如何看待这些争议呢？很期待听到您的见解。"
]
sentences = [
    "I don't think the U.S. successful moon landing is a conspiracy theory.",
    "I think the U.S. successful moon landing is a conspiracy theory.",
    "I hold a neutral view on whether the successful U.S. moon landing was a conspiracy theory.",
    ]
embeddings = model.encode(sentences)
s=model.encode(["I believe the U.S. successful moon landing was a genuine historical achievement. This conclusion is supported by a wealth of public evidence: the analysis of lunar samples by scientific institutions worldwide, independently tracked lunar orbiter data, and photographs of the Apollo mission relics captured by probes from multiple countries in recent years.Furthermore, such a massive project involving hundreds of thousands of scientists and engineers could hardly be concealed for long, and the Soviet Union, as a Cold War rival, never questioned its authenticity through its monitoring. These facts together provide solid scientific and historical evidence."])
similarities = model.similarity(s, embeddings)
print(similarities)
# [4, 4]
'''
nlp apikey
https://docs.aimlapi.com/use-cases/find-relevant-answers-semantic-search-with-text-embeddings
5df668b232764e17ac69a5a7518ca6db
'''