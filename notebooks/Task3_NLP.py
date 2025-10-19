"""
Task 3: NLP with spaCy
Goal: Named Entity Recognition and Sentiment Analysis
"""

import spacy
from spacy import displacy
import pandas as pd
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

print("="*70)
print("NLP WITH SPACY: NER & SENTIMENT ANALYSIS")
print("="*70)

# Sample Amazon-style reviews
reviews = [
    "I bought the Apple iPhone 14 Pro from Amazon and it's amazing! The camera quality is superb.",
    "Samsung Galaxy S23 is terrible. Battery life is awful and the screen broke after 2 weeks.",
    "The Sony WH-1000XM5 headphones are the best I've ever used. Worth every penny!",
    "Nike Air Max shoes arrived damaged. Very disappointed with the quality.",
    "My Dell XPS 15 laptop is fantastic for coding. Intel processor is super fast!",
    "Microsoft Surface Pro is overpriced garbage. Keyboard doesn't work properly.",
    "Love my new iPad Pro from Apple! Great for drawing and note-taking.",
    "The Bose QuietComfort earbuds have excellent noise cancellation. Highly recommend!",
    "Canon EOS R5 camera is professional grade. Perfect for photography enthusiasts.",
    "Terrible experience with HP Pavilion. Customer service was unhelpful and rude."
]

print(f"\n📝 Analyzing {len(reviews)} product reviews...\n")

# Task 3.1: Named Entity Recognition
print("[TASK 3.1] Named Entity Recognition\n")
print("-" * 70)

all_entities = []
for i, review in enumerate(reviews, 1):
    doc = nlp(review)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    print(f"\nReview {i}: '{review[:60]}...'")
    if entities:
        print("Entities found:")
        for entity, label in entities:
            print(f"  - {entity:20s} → {label}")
            all_entities.append((entity, label))
    else:
        print("  No entities found")

# Count entity types
entity_types = Counter([label for _, label in all_entities])
print("\n" + "="*70)
print("📊 Entity Type Summary:")
for entity_type, count in entity_types.most_common():
    print(f"  {entity_type:15s}: {count}")

# Extract products and brands
products = [ent for ent, label in all_entities if label in ['PRODUCT', 'ORG', 'PERSON']]
print(f"\n🏷️  Unique Products/Brands Mentioned: {len(set(products))}")
for product in sorted(set(products)):
    print(f"  • {product}")

# Task 3.2: Rule-Based Sentiment Analysis
print("\n" + "="*70)
print("[TASK 3.2] Sentiment Analysis (Rule-Based)")
print("-" * 70)

# Positive and negative keywords
positive_words = ['amazing', 'superb', 'best', 'fantastic', 'excellent', 'love', 
                  'great', 'worth', 'recommend', 'perfect', 'professional']
negative_words = ['terrible', 'awful', 'broke', 'damaged', 'disappointed', 
                  'garbage', 'overpriced', 'unhelpful', 'rude', 'worst']

sentiments = []

for i, review in enumerate(reviews, 1):
    doc = nlp(review.lower())
    tokens = [token.text for token in doc if not token.is_punct]
    
    # Count sentiment words
    pos_count = sum(1 for word in tokens if word in positive_words)
    neg_count = sum(1 for word in tokens if word in negative_words)
    
    # Determine sentiment
    if pos_count > neg_count:
        sentiment = "POSITIVE"
        score = pos_count
    elif neg_count > pos_count:
        sentiment = "NEGATIVE"
        score = neg_count
    else:
        sentiment = "NEUTRAL"
        score = 0
    
    sentiments.append(sentiment)
    
    print(f"\nReview {i}: '{review[:50]}...'")
    print(f"  Sentiment: {sentiment} (Positive: {pos_count}, Negative: {neg_count})")

# Summary
sentiment_counts = Counter(sentiments)
print("\n" + "="*70)
print("📊 Sentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(reviews)) * 100
    print(f"  {sentiment:10s}: {count} ({percentage:.1f}%)")

# Create results DataFrame
results_df = pd.DataFrame({
    'Review': [r[:50] + '...' for r in reviews],
    'Sentiment': sentiments,
    'Length': [len(r.split()) for r in reviews]
})

print("\n📋 Results Table:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('outputs/nlp_results.csv', index=False)
print("\n✓ Results saved to 'outputs/nlp_results.csv'")

# Save detailed entity report
with open('outputs/nlp_entity_report.txt', 'w') as f:
    f.write("NAMED ENTITY RECOGNITION REPORT\n")
    f.write("="*70 + "\n\n")
    for i, review in enumerate(reviews, 1):
        doc = nlp(review)
        f.write(f"Review {i}: {review}\n")
        f.write(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}\n\n")

print("✓ Entity report saved to 'outputs/nlp_entity_report.txt'")

print("\n" + "="*70)
print("TASK 3 COMPLETE ✓")
print("="*70)