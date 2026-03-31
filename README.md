# CS Career Intelligence from Reddit

Computer science students frequently rely on Reddit communities such as r/csMajors and r/cscareerquestions for advice on internships, recruiting strategies, course selection, and the broader tech job market. These discussions contain valuable insights from students and professionals, but the information is scattered across thousands of posts and comment threads, making it difficult to identify overall trends or actionable advice.

This project aims to build a big data pipeline that collects and analyzes Reddit discussions related to CS careers and internship preparation. The system will automatically ingest posts and comments from relevant subreddits, process the text using PySpark, and extract structured insights about common topics and trends in career discussions.

While some categories of discussion are expected (such as internships, interviews, or course selection), the analysis will not rely solely on predefined labels. Instead, the pipeline will detect recurring themes directly from the data, meaning that some insights shown in the final results may vary depending on patterns present in the collected dataset.

To simulate a continuously updating system, the pipeline will periodically retrieve new Reddit posts through the Reddit API, allowing the analysis to update regularly in a near real-time manner.
