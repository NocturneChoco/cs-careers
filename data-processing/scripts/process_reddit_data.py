from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, regexp_replace, split, explode, count, desc,
    from_unixtime, year, month, dayofmonth, hour, length,
    when, trim, udf, collect_list, avg, sum as spark_sum
)
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline
import re
import json

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Reddit CS Career Analysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# Set log level to reduce verbosity
spark.sparkContext.setLogLevel("WARN")

print("=" * 80)
print("REDDIT CS CAREER DATA PROCESSING PIPELINE")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n[STEP 1] Loading raw Reddit data...")

# Load the JSON data
df = spark.read.json("data/raw_posts.json")

print(f"✓ Loaded {df.count()} posts")
print("\nData Schema:")
df.printSchema()

# ============================================================================
# 2. DATA CLEANING & ENRICHMENT
# ============================================================================
print("\n[STEP 2] Cleaning and enriching data...")

# Convert Unix timestamp to readable datetime
df_cleaned = df.withColumn(
    "created_datetime",
    from_unixtime(col("created_utc"))
)

# Extract date components for time-based analysis
df_cleaned = df_cleaned \
    .withColumn("year", year(col("created_datetime"))) \
    .withColumn("month", month(col("created_datetime"))) \
    .withColumn("day", dayofmonth(col("created_datetime"))) \
    .withColumn("hour", hour(col("created_datetime")))

# Combine title and selftext for complete content analysis
df_cleaned = df_cleaned.withColumn(
    "full_text",
    when(col("selftext").isNull() | (col("selftext") == ""), col("title"))
    .otherwise(regexp_replace(col("title") + " " + col("selftext"), r"\s+", " "))
)

# Calculate text length
df_cleaned = df_cleaned.withColumn(
    "text_length",
    length(col("full_text"))
)

# Categorize post engagement
df_cleaned = df_cleaned.withColumn(
    "engagement_level",
    when(col("num_comments") == 0, "No Engagement")
    .when(col("num_comments") <= 2, "Low Engagement")
    .when(col("num_comments") <= 10, "Medium Engagement")
    .otherwise("High Engagement")
)

# Identify if post contains a URL/image
df_cleaned = df_cleaned.withColumn(
    "has_media",
    when(col("url").contains("i.redd.it") | col("url").contains("imgur"), "Image")
    .when(col("url").contains("reddit.com/r/"), "Text Post")
    .otherwise("External Link")
)

print(f"✓ Cleaned and enriched {df_cleaned.count()} posts")

# ============================================================================
# 3. TEXT PREPROCESSING
# ============================================================================
print("\n[STEP 3] Preprocessing text for analysis...")

# Clean text: lowercase, remove special characters, URLs
df_text = df_cleaned.withColumn(
    "cleaned_text",
    lower(regexp_replace(
        regexp_replace(
            regexp_replace(col("full_text"), r"http\S+", ""),  # Remove URLs
            r"[^a-zA-Z\s]", " "  # Remove special chars
        ),
        r"\s+", " "  # Normalize whitespace
    ))
)

# Tokenization
tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")
df_tokenized = tokenizer.transform(df_text)

# Remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = remover.transform(df_tokenized)

# Filter out very short posts (less than 3 words)
df_filtered = df_filtered.filter(length(col("cleaned_text")) > 10)

print(f"✓ Preprocessed {df_filtered.count()} posts")

# ============================================================================
# 4. KEYWORD & TOPIC EXTRACTION
# ============================================================================
print("\n[STEP 4] Extracting keywords and topics...")

# Explode words for word frequency analysis
df_words = df_filtered.select(
    "id", "title", "score", "num_comments", "created_datetime",
    explode(col("filtered_words")).alias("word")
)

# Calculate word frequencies
word_freq = df_words.groupBy("word") \
    .agg(count("*").alias("frequency")) \
    .filter(length(col("word")) > 3) \
    .orderBy(desc("frequency"))

print("\n📊 Top 20 Keywords:")
word_freq.show(20, truncate=False)

# Define career-related keyword categories
def categorize_topic(text):
    text_lower = text.lower()
    
    categories = []
    
    # Internship related
    if any(word in text_lower for word in ['internship', 'intern', 'summer', 'co-op', 'coop']):
        categories.append('Internship')
    
    # Job search related
    if any(word in text_lower for word in ['job', 'application', 'apply', 'applying', 'offer', 'search']):
        categories.append('Job Search')
    
    # Interview related
    if any(word in text_lower for word in ['interview', 'leetcode', 'oa', 'assessment', 'coding challenge']):
        categories.append('Interview Prep')
    
    # Resume/career advice
    if any(word in text_lower for word in ['resume', 'cv', 'portfolio', 'project', 'experience']):
        categories.append('Resume/Profile')
    
    # Salary/compensation
    if any(word in text_lower for word in ['salary', 'compensation', 'pay', 'tc', 'offer']):
        categories.append('Compensation')
    
    # Company specific
    if any(word in text_lower for word in ['amazon', 'google', 'meta', 'microsoft', 'faang', 'big tech']):
        categories.append('Big Tech')
    
    # Education
    if any(word in text_lower for word in ['course', 'class', 'degree', 'major', 'masters', 'university', 'college']):
        categories.append('Education')
    
    # Career advice/guidance
    if any(word in text_lower for word in ['advice', 'help', 'question', 'should i', 'how to', 'tips']):
        categories.append('Seeking Advice')
    
    # Rejection/struggle
    if any(word in text_lower for word in ['reject', 'ghost', 'fail', 'struggle', 'difficult', 'hard']):
        categories.append('Challenges')
    
    return categories if categories else ['General']

categorize_udf = udf(categorize_topic, ArrayType(StringType()))

df_categorized = df_filtered.withColumn(
    "topics",
    categorize_udf(col("full_text"))
)

# Explode topics for counting
df_topics = df_categorized.select(
    "id", "title", "score", "num_comments",
    explode(col("topics")).alias("topic")
)

topic_distribution = df_topics.groupBy("topic") \
    .agg(
        count("*").alias("post_count"),
        avg("score").alias("avg_score"),
        avg("num_comments").alias("avg_comments")
    ) \
    .orderBy(desc("post_count"))

print("\n📋 Topic Distribution:")
topic_distribution.show(truncate=False)

# ============================================================================
# 5. TF-IDF ANALYSIS
# ============================================================================
print("\n[STEP 5] Running TF-IDF analysis...")

# CountVectorizer to get term frequencies
cv = CountVectorizer(inputCol="filtered_words", outputCol="raw_features", vocabSize=500)
cv_model = cv.fit(df_filtered)
df_cv = cv_model.transform(df_filtered)

# IDF to get importance weights
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_cv)
df_tfidf = idf_model.transform(df_cv)

# Get vocabulary
vocabulary = cv_model.vocabulary

print(f"✓ Extracted {len(vocabulary)} unique terms")
print(f"\nSample vocabulary: {vocabulary[:20]}")

# ============================================================================
# 6. ENGAGEMENT ANALYSIS
# ============================================================================
print("\n[STEP 6] Analyzing post engagement patterns...")

# Engagement by topic
engagement_by_topic = df_topics.groupBy("topic") \
    .agg(
        avg("score").alias("avg_upvotes"),
        avg("num_comments").alias("avg_comments"),
        count("*").alias("total_posts")
    ) \
    .orderBy(desc("avg_comments"))

print("\n💬 Engagement by Topic:")
engagement_by_topic.show(truncate=False)

# Engagement by hour of day
engagement_by_hour = df_cleaned.groupBy("hour") \
    .agg(
        count("*").alias("post_count"),
        avg("score").alias("avg_score"),
        avg("num_comments").alias("avg_comments")
    ) \
    .orderBy("hour")

print("\n🕐 Engagement by Hour of Day:")
engagement_by_hour.show(24, truncate=False)

# ============================================================================
# 7. COMPANY MENTIONS EXTRACTION
# ============================================================================
print("\n[STEP 7] Extracting company mentions...")

companies = ['amazon', 'google', 'meta', 'facebook', 'microsoft', 'apple', 
             'netflix', 'uber', 'lyft', 'airbnb', 'salesforce', 'oracle',
             'ibm', 'intel', 'nvidia', 'tesla', 'spotify', 'twitter', 'x corp']

def extract_companies(text):
    if not text:
        return []
    text_lower = text.lower()
    found = [company for company in companies if company in text_lower]
    return found if found else []

extract_companies_udf = udf(extract_companies, ArrayType(StringType()))

df_companies = df_filtered.withColumn(
    "mentioned_companies",
    extract_companies_udf(col("full_text"))
).filter(length(col("mentioned_companies")) > 0)

df_company_mentions = df_companies.select(
    explode(col("mentioned_companies")).alias("company"),
    "score",
    "num_comments"
)

company_stats = df_company_mentions.groupBy("company") \
    .agg(
        count("*").alias("mention_count"),
        avg("score").alias("avg_score"),
        avg("num_comments").alias("avg_engagement")
    ) \
    .orderBy(desc("mention_count"))

print("\n🏢 Company Mentions:")
company_stats.show(truncate=False)

# ============================================================================
# 8. SENTIMENT INDICATORS
# ============================================================================
print("\n[STEP 8] Analyzing sentiment indicators...")

# Simple sentiment based on keywords
def get_sentiment(text):
    if not text:
        return "Neutral"
    
    text_lower = text.lower()
    
    positive_words = ['success', 'offer', 'accepted', 'great', 'excited', 'happy', 
                      'amazing', 'awesome', 'love', 'best', 'good', 'congratulations']
    negative_words = ['reject', 'ghost', 'fail', 'sad', 'depressed', 'anxious', 
                      'worried', 'stress', 'difficult', 'hard', 'no response', 'bad']
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Neutral"

sentiment_udf = udf(get_sentiment, StringType())

df_sentiment = df_filtered.withColumn(
    "sentiment",
    sentiment_udf(col("full_text"))
)

sentiment_dist = df_sentiment.groupBy("sentiment") \
    .agg(
        count("*").alias("count"),
        avg("score").alias("avg_score"),
        avg("num_comments").alias("avg_comments")
    ) \
    .orderBy(desc("count"))

print("\n😊 Sentiment Distribution:")
sentiment_dist.show(truncate=False)

# ============================================================================
# 9. SAVE PROCESSED DATA
# ============================================================================
print("\n[STEP 9] Saving processed data...")

# Save topic analysis
topic_distribution.coalesce(1).write.mode("overwrite").json("data-processing/topic_analysis")
print("✓ Saved topic_analysis")

# Save word frequencies
word_freq.limit(200).coalesce(1).write.mode("overwrite").json("data-processing/word_frequencies")
print("✓ Saved word_frequencies")

# Save engagement analysis
engagement_by_topic.coalesce(1).write.mode("overwrite").json("data-processing/engagement_by_topic")
engagement_by_hour.coalesce(1).write.mode("overwrite").json("data-processing/engagement_by_hour")
print("✓ Saved engagement analyses")

# Save company mentions
company_stats.coalesce(1).write.mode("overwrite").json("data-processing/company_mentions")
print("✓ Saved company_mentions")

# Save sentiment analysis
sentiment_dist.coalesce(1).write.mode("overwrite").json("data-processing/sentiment_analysis")
print("✓ Saved sentiment_analysis")

# Save full processed dataset
df_categorized.select(
    "id", "title", "full_text", "score", "num_comments",
    "created_datetime", "year", "month", "day", "hour",
    "engagement_level", "has_media", "text_length", "topics"
).coalesce(1).write.mode("overwrite").json("data-processing/processed_posts")
print("✓ Saved processed_posts")

# ============================================================================
# 10. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("PROCESSING COMPLETE - SUMMARY STATISTICS")
print("=" * 80)

print(f"\n📊 Total Posts Processed: {df_filtered.count()}")
print(f"📅 Date Range: {df_cleaned.agg({'created_datetime': 'min'}).collect()[0][0]} to {df_cleaned.agg({'created_datetime': 'max'}).collect()[0][0]}")
print(f"💬 Total Comments: {df_cleaned.agg({'num_comments': 'sum'}).collect()[0][0]}")
print(f"⬆️  Total Upvotes: {df_cleaned.agg({'score': 'sum'}).collect()[0][0]}")
print(f"📝 Average Post Length: {df_cleaned.agg({'text_length': 'avg'}).collect()[0][0]:.0f} characters")

print("\n✅ All processing complete! Check data-processing/ folder for outputs.")

# Stop Spark session
spark.stop()