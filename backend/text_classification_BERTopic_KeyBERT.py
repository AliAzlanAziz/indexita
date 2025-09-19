import time
import numpy as np
from tqdm import trange
import torch
import spacy

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
# -----------------------------
# Config
# -----------------------------
PREFERRED_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",   # better quality, heavier
    "sentence-transformers/all-MiniLM-L6-v2",    # lighter, very fast
]
DEFAULT_BATCH_MPNet = 12
DEFAULT_BATCH_MiniLM = 32

nlp = spacy.load("en_core_web_sm", disable=["ner"])  # keep parser ON
ALLOWED_POS = {"NOUN", "PROPN", "VERB", "ADJ"}

# -----------------------------
# Utilities
# -----------------------------
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_st_model_with_fallback():
    """
    Try a high-quality model first; if OOM, fall back to a smaller model.
    Returns: (model, model_name, batch_size_hint)
    """
    device = get_device()
    for name in PREFERRED_MODELS:
        try:
            model = SentenceTransformer(name, device=device)
            # batch size hint by model
            if "mpnet" in name:
                return model, name, DEFAULT_BATCH_MPNet
            else:
                return model, name, DEFAULT_BATCH_MiniLM
        except RuntimeError as e:
            # Rare, but handle immediate OOM on load
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            else:
                raise
    raise RuntimeError("Failed to load any SentenceTransformer model.")

def spacy_pos_tokenizer(text: str):
    doc = nlp(text)
    return [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and t.pos_ in ALLOWED_POS and len(t) >= 3]

def encode_in_chunks(model, texts, batch_size, normalize=True):
    """
    Encode texts in safe batches, adaptively shrinking batch_size on OOM.
    """
    embeddings = []
    i = 0
    while i < len(texts):
        bs = min(batch_size, len(texts) - i)
        try:
            batch = texts[i:i+bs]
            embs = model.encode(
                batch,
                batch_size=bs,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(embs)
            i += bs
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                # back off batch size
                if bs == 1:
                    # fall back to CPU if even a single item fails (very unlikely)
                    model = SentenceTransformer(model.model_name, device="cpu")
                batch_size = max(1, bs // 2)
                continue
            else:
                raise
    return np.vstack(embeddings)

def build_bertopic(n_docs, embedding_model, umap_n_components=5, umap_n_neighbors=15, min_cluster_size=15):
    # Clamp to avoid small-N failures
    n_components = max(2, min(umap_n_components, n_docs - 2))
    n_neighbors  = max(2, min(umap_n_neighbors,  n_docs - 1))
    # For small corpora, many useful terms are one-offs
    vectorizer_model = CountVectorizer(
        tokenizer=spacy_pos_tokenizer,
        ngram_range=(1, 2),
        min_df=1,              # was 2; allow rare but meaningful terms
        max_df=0.95,
        max_features=20000
    )
    representation_model = KeyBERTInspired(top_n_words=20)

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        init="random",         # <— bypass spectral eigendecomp that caused k>=N
        low_memory=True,
        random_state=42,
        min_dist=0.0,
    )
    hdbscan_model = HDBSCAN(
        # scale down for small corpora so we actually form clusters
        min_cluster_size=max(2, min(min_cluster_size, max(2, n_docs // 3))),
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True
    )
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=True,
        verbose=True,
        low_memory=True
    )

def fit_bertopic_on_embeddings(topic_model, docs, embeddings):
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    return topic_model, topics, probs

def keybert_for_doc(doc, kw_model, topic_words=None, top_n=5, use_mmr=True, diversity=0.4):
    """
    Extract per-document tags. If topic_words are provided (from BERTopic),
    restrict candidates to keep tags aligned with the global topic.
    """
    candidates = None
    if topic_words:
        # Use only the top-N topic words as candidates; keep them clean and unique.
        candidates = list({w for w in topic_words})
    # For short texts, limit n-grams to (1,2) to avoid noisy long phrases.
    results = kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_mmr=use_mmr,
        diversity=diversity,
        top_n=top_n,
        candidates=candidates
    )
    # results is list of tuples (key, score)
    return [k for k, _ in results]

def hybrid_themes_and_tags(docs, topn_topic_words=15, per_doc_tags=5):
    """
    Full pipeline:
      1) embed (GPU if available) with robust model + chunking
      2) BERTopic for global themes
      3) KeyBERT for per-document tags (restricted by each doc's assigned topic terms)
    """
    # 1) Load embedding model with fallback
    st_model, model_name, batch_hint = load_st_model_with_fallback()
    print(f"[Embedding model] {model_name} on {get_device()} (batch≈{batch_hint})")

    # 2) Chunked embedding
    print("[Embedding] Encoding documents...")
    embeddings = encode_in_chunks(st_model, docs, batch_size=batch_hint, normalize=True)
    print(f"[Embedding] Done. Shape: {embeddings.shape}")

    # 3) BERTopic (using precomputed embeddings)
    print("[BERTopic] Fitting topic model...")
    topic_model = build_bertopic(len(docs), embedding_model=st_model)
    topic_model, topics, probs = fit_bertopic_on_embeddings(topic_model, docs, embeddings)
    print("[BERTopic] Done.")

    # 4) Prepare KeyBERT using the SAME embedding model instance (saves memory & consistent space)
    kw_model = KeyBERT(model=st_model)

    # 5) Per-document tags aligned with the doc's topic
    doc_tags = []
    for idx, doc in enumerate(docs):
        topic_id = topics[idx]
        topic_words = None
        if topic_id != -1:
            # get_topic returns [(word, weight), ...]
            topic_terms = topic_model.get_topic(topic_id) or []
            topic_words = [w for (w, _) in topic_terms[:topn_topic_words]]
        tags = keybert_for_doc(
            doc, kw_model, topic_words=topic_words, top_n=per_doc_tags, use_mmr=True, diversity=0.4
        )
        doc_tags.append(tags)

    return {
        "topic_model": topic_model,
        "topics": topics,
        "topic_probabilities": probs,
        "per_doc_tags": doc_tags,
        "embedding_model_name": model_name,
    }

def extract_candidates_spacy(text: str, topic_words=None, max_len=4):
    doc = nlp(text)
    phrases = set()

    # Only use noun_chunks when parser is available
    if "parser" in nlp.pipe_names:
        for chunk in doc.noun_chunks:
            toks = [t.lemma_.lower() for t in chunk if t.is_alpha and not t.is_stop]
            if toks and 1 <= len(toks) <= max_len:
                phrases.add(" ".join(toks))

    # Noun phrases (cleaned, lemmatized)
    for chunk in doc.noun_chunks:
        toks = [
            t.lemma_.lower() for t in chunk
            if t.is_alpha and not t.is_stop
        ]
        if toks:
            phrase = " ".join(toks)
            if 1 <= len(phrase.split()) <= max_len:
                phrases.add(phrase)

    # Content verbs & standalone content words
    verbs = {t.lemma_.lower() for t in doc if t.pos_ == "VERB" and t.is_alpha and not t.is_stop}
    content = {
        t.lemma_.lower()
        for t in doc
        if t.is_alpha and not t.is_stop and t.pos_ in ALLOWED_POS and len(t) >= 3
    }

    if topic_words:
        phrases |= {w for w in topic_words if len(w) >= 3}

    return list(phrases | verbs | content)

def keybert_for_doc(doc, kw_model, topic_words=None, top_n=5, use_mmr=True, diversity=0.6):
    candidates = extract_candidates_spacy(doc, topic_words)
    if not candidates:
        candidates = None  # let KeyBERT generate its own if our list is empty

    results = kw_model.extract_keywords(
        doc, candidates=candidates, keyphrase_ngram_range=(1, 4),
        stop_words=None, use_mmr=False,  # or keep True but set diversity=0.2
        top_n=top_n,
    )
    return [k for k, _ in results]

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Replace with your corpus
    docs = [
        "I need to buy coffee immediately because my supply at home ran out overnight, and without a strong cup in the morning my productivity tanks. I am comparing same-day delivery options from local roasters versus marketplace sellers, weighing freshness against speed, shipping fees, and reliability. A medium roast with chocolate and nut notes is ideal, but I’ll take a light roast if it arrives sooner. Subscriptions look tempting for convenience, yet I’m unsure about commitment and grind settings for my brewer.",
        "I’m researching an entry-level espresso setup that won’t waste beans while I learn. Everyone says the grinder matters more than the machine, so I’m prioritizing a consistent burr grinder before picking between a single-boiler or heat-exchanger. I want a 58 mm portafilter for accessory compatibility, a reliable PID for temperature stability, and a fast warm-up. My goal is repeatable shots with a balanced extraction, syrupy mouthfeel, and room to grow into milk drinks later.",
        "Instant coffee is undeniably convenient when deadlines pile up, yet the flavor often tastes thin or bitter. I’m exploring specialty instant packets made from high-quality beans, freeze-dried to preserve aromatics, so I can keep sachets in my bag for travel or late nights. If the taste approaches a decent pour-over and the caffeine hits smoothly, paying a premium could be worth it. I also care about clear labeling for origin, roast date equivalents, and ethical sourcing claims.",
        "For pour-over, I’m choosing between a V60, Kalita Wave, and flat-bottom drippers, trying to understand how geometry affects flow rate, agitation, and extraction balance. I plan to use a gooseneck kettle to control the pour, test pulse pours against continuous pouring, and track total brew time. With a light roast washed Ethiopian, I’m targeting juicy acidity and floral aromatics. Consistency will come from weighing doses, rinsing filters, and logging grind settings for each attempt.",
        "My French press routine needs refinement because I’m getting sludge and muddled flavors. I’ll switch to a coarser grind, use a proper coffee-to-water ratio, and skim the crust before plunging to reduce fines. Letting the brew settle for a minute after pressing might clarify the cup. If I still taste bitterness, I’ll shorten the steep time or lower the water temperature. A metal mesh filter is durable, but I’m curious whether using a paper filter hack can clean up the body.",
        "Cold brew concentrate seems ideal for hot afternoons and batch preparation. I’m planning a 1:5 ratio with a medium-coarse grind, steeped for 14–18 hours in the fridge, then diluted over ice. I’ll experiment with different origins to see which maintain sweetness without turning flat. Storage in an airtight container should keep it fresh for several days. If the concentrate proves versatile, I’ll try nitro style at home with a whipped-cream canister for a creamy texture.",
        "Milk texturing is my weak point. I can pull a decent espresso shot, but my microfoam either splits or becomes too airy, so latte art fails. I’ll practice with a larger pitcher for better whirlpool control, stop stretching at around 30–40°C, and finish near 55–60°C to preserve sweetness. A thermometer will keep me honest while I learn. Once the foam integrates like wet paint, I’ll start working on simple hearts and tulips before moving to rosettas.",
        "I’m evaluating decaf options for late-evening cravings when I still want the comfort of coffee without the jitters. The Swiss Water Process sounds appealing because it avoids chemical solvents and can preserve more flavor. I’ll look for roasters who publish cupping notes for their decaf offerings rather than treating them as an afterthought. If I can get caramelized sweetness and a clean finish in milk drinks, decaf cappuccinos could become a nightly ritual.",
        "Single-origin coffees help me learn how terroir shapes flavor. A washed Ethiopian might deliver jasmine and citrus with a tea-like body, while a natural Brazilian can bring cocoa and ripe fruit with heavier sweetness. Blends provide balance and consistency, which is great for espresso dialing-in, but I enjoy the seasonal variety of limited lots. I plan to record tasting notes, extraction times, and grinder clicks to connect variables with what I taste in the cup.",
        "Roast level dramatically changes perception. Light roasts preserve origin character but can taste underdeveloped if my extraction isn’t even; dark roasts mask nuance but offer comforting bitterness and heavy body. I’m aiming for a medium light profile that brightens milk drinks without becoming sour. If the beans show oil on the surface too soon, staling accelerates. I’ll buy smaller bags more frequently to keep the flavor lively and avoid flat, papery notes.",
        "Freshness matters, yet I learned that brewing too soon after roast can be gassy and erratic. I’ll target a sweet spot four to fourteen days off roast, depending on the style and my storage method. One-way valve bags help, but I’m transferring to airtight containers and keeping them away from heat and light. Freezing in single-dose jars might preserve volatile aromatics when I stock up during sales, as long as I grind from frozen to reduce condensation.",
        "Choosing a grinder forces me to think about workflow and particle distribution. Hand grinders are quiet, portable, and excellent for pour-over, but electric options save time for morning espresso. Conical burrs can produce sweet, forgiving cups, while flat burrs often give clarity and separation of flavors at the cost of more retention. I’ll clean the burr chamber regularly and weigh both input and output to track consistency and minimize waste.",
        "Water quality is the invisible ingredient. If minerals are off, extraction stalls or over-accelerates, causing sourness or harsh bitterness even when everything else is correct. I’m looking at SCA-aligned water recipes that balance bicarbonate and hardness to protect equipment while optimizing flavor. Filtered tap water could be a baseline, but a simple DIY mineral mix may be more consistent. I’ll standardize on one recipe so dialing in becomes repeatable across beans.",
        "Caffeine tolerance creeps up on me, so I’m auditing how many shots I consume before noon and whether it affects sleep. Instead of chasing intensity, I want clarity and sweetness, which often means precise grinding and patient pouring rather than bigger doses. Keeping a small cup size helps me savor without overdoing it. On heavy workdays I’ll alternate with herbal tea or half-caf blends to avoid that afternoon crash and evening restlessness.",
        "For travel, I want a compact kit that still brews a satisfying cup: an AeroPress Go, a small hand grinder, folding scale, and a collapsible kettle. Pre-weighed filter packs save time in hotel rooms and Airbnbs. I’ll bring a reusable tumbler to cut waste and a zip pouch for wet filters. If I can keep the routine simple and repeatable, I’ll stop relying on random café quality and maintain energy during early flights or long road trips.",
        "Sustainability matters in my purchasing decisions. I’m seeking roasters who publish farm relationships, pay premiums above commodity prices, and invest in long-term quality. Packaging that’s recyclable or compostable is a plus, as is carbon-neutral shipping. At home, I’ll compost spent grounds or use them for gardening, and I’ll descale my machine with eco-friendly products. If the coffee tastes great and the supply chain is transparent, I’m happy to support it regularly.",
        "I’m building a coffee budget that balances quality with volume. Grocery-store beans are affordable but often stale; boutique roasters taste better but strain the wallet. Subscription discounts and larger bag sizes can help if I brew consistently. I’ll track cost per cup against flavor enjoyment to decide where premium is worth it. When experimenting with new origins, sampler packs minimize risk and help me discover profiles I actually finish rather than abandon.",
        "Ordering online introduces logistics I can’t ignore: estimated roast dates, packing lead times, carrier reliability, and delivery windows that may miss my morning routine. I prefer vendors that roast to order and ship within twenty-four hours, include clear batch labeling, and accept returns if a bag arrives damaged. Transparent inventory avoids bait-and-switch substitutions. If a shop nails freshness, communication, and flavor, I’ll stick with them and recommend widely."
    ]

    start = time.perf_counter()
    results = hybrid_themes_and_tags(docs, topn_topic_words=30, per_doc_tags=10)
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {elapsed:.3f} s ({elapsed/60:.3f} min)")

    topic_model = results["topic_model"]
    topics = results["topics"]
    per_doc_tags = results["per_doc_tags"]

    # Show global topics
    print("\n=== Global Topics (top 5 words each) ===")
    for topic_id in sorted(set(t for t in topics if t != -1)):
        terms = topic_model.get_topic(topic_id) or []
        words = ", ".join([w for w, _ in terms[:10]])
        print(f"Topic {topic_id}: {words}")

    # Show per-document tags
    print("\n=== Per-document tags ===")
    for i, (doc, tags) in enumerate(zip(docs, per_doc_tags)):
        print(f"[{i}] {doc}\n     -> {tags}")
