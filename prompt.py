import clip
import torch
import nltk
nltk.data.path.append('/mnt/nltk_data')
from nltk.corpus import wordnet as wn

# 测试是否能成功加载WordNet数据
print(wn.synsets('dog'))
# 假设我们已经有一个预训练的CLIP模型和处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def query_similar_categories(target_category):
    # 模拟查询类似类别的函数，实际实现可能涉及到一个数据库或API调用
    similar_categories = {
        "云豹": ["猎豹", "豹子"],
        "forest": ["woodland", "wood"]
    }
    return similar_categories.get(target_category, [])

def get_synonyms(word):
    # 使用WordNet查找同义词
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def create_clip_prompts(target_category):
    # 获取类似类别并形成提示
    similar_categories = query_similar_categories(target_category)
    similar_prompts = [f"a {target_category} similar to {category}" for category in similar_categories]

    # 获取同义词并形成提示
    synonyms = get_synonyms(target_category)
    synonym_prompts = [f"a photo of {synonym}" for synonym in synonyms]

    return similar_prompts, synonym_prompts

def encode_prompts(prompts):
    # 使用CLIP文本编码器处理提示
    text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    return text_features

def main():
    target_category = "forest"

    # 创建CLIP提示
    similar_prompts, synonym_prompts = create_clip_prompts(target_category)

    # 编码提示
    similar_features = encode_prompts(similar_prompts)
    synonym_features = encode_prompts(synonym_prompts)

    print("Similar Prompts Features:", similar_features)
    print("Synonym Prompts Features:", synonym_features)

if __name__ == "__main__":
    main()
