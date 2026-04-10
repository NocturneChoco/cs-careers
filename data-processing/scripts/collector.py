import requests
import time
import json
import boto3

HEADERS = {"User-Agent": "cs-career-intel/1.0 (course project)"}

def fetch_posts(subreddit, listing="new", limit_per_page=100, max_pages=10):
    posts = []
    after = None
    
    for _ in range(max_pages):
        url = f"https://www.reddit.com/r/{subreddit}/{listing}.json"
        params = {"limit": limit_per_page}
        if after:
            params["after"] = after
        
        res = requests.get(url, headers=HEADERS, params=params)
        
        if res.status_code != 200:
            print(f"Error: {res.status_code}")
            break
        
        data = res.json()["data"]
        batch = data["children"]
        
        if not batch:
            break
        
        for item in batch:
            p = item["data"]
            posts.append({
                "id": p["id"],
                "title": p["title"],
                "selftext": p["selftext"],
                "score": p["score"],
                "created_utc": p["created_utc"],
                "num_comments": p["num_comments"],
                "url": p["url"],
            })
        
        after = data["after"]
        if not after:
            break
        
        time.sleep(1)  # 别打太快，礼貌一点
    
    return posts

# 抓多个 subreddit 和 listing 类型
all_posts = []
for sub in ["csMajors", "cscareerquestions"]:
    for listing in ["new", "top"]:
        print(f"Fetching r/{sub}/{listing}...")
        batch = fetch_posts(sub, listing=listing, max_pages=10)
        all_posts.extend(batch)
        time.sleep(2)

print(f"Total posts: {len(all_posts)}")

# 存到本地 / 上传 S3
with open("raw_posts.json", "w") as f:
    json.dump(all_posts, f)