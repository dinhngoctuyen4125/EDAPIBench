import requests
import json
import urllib.parse
from pathlib import Path
import logging
from typing import List
from tqdm import tqdm
import os
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
sys.path.append(os.getcwd())
from utils import init_log

SG_TOKEN = os.environ["SG_TOKEN"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

# Libraries to search
# LIBs = [
    # "transformers", "tensorflow", "pytorch", "numpy", "pandas", "scipy", "sklearn", "seaborn"
# ]
LIBs = ["numpy"]  # NumPy pilot run.

ALIAS = {
    "tensorflow": ["tf"],
    "numpy": ["np"],
    "pandas": ["pd"],
    "pandas.DataFrame": ["df"],
    "sklearn": ["sk"],
    "scipy.stats": ['sts','st'],
    "scipy.special":['sps', 'sp'],
    "scipy.linalg":['spl'],
    "scipy": ["sc", "sp"],
    "seaborn": ["sn", "sns", "sb"],
}

# Search configuration
# MAX_COUNT = 20000  # Max total results
MAX_COUNT = 20  # Candidate files per replacement API; not the final sample count.
# GITHUB_MAX_PAGE = 10  # Max pages for GitHub
GITHUB_MAX_PAGE = 1  # One page per query for the pilot run.
TIMEOUT = 30  # Base request timeout in seconds

# URL templates
# SOURCEGRAPH_SEARCH_URL = "https://sourcegraph.com/.api/search/stream?q=context:global {keywords} timeout:45s count:50000 type:file lang:python &v=V3&t=keyword&sm=0&cm=t&max-line-len=5120"
SOURCEGRAPH_SEARCH_URL = "https://sourcegraph.com/.api/search/stream?q=context:global {keywords} timeout:45s count:{max_count} type:file lang:python &v=V3&t=keyword&sm=0&cm=t&max-line-len=5120"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{file}"
GITHUB_SEARCH_BASE_URL = "https://api.github.com/search/code"

# Request headers
SOURCEGRAPH_HEADERS = {
    'Accept': 'application/json',
    'Authorization': f'token {SG_TOKEN}'
}
GITHUB_HEADERS = {
    'Accept': 'application/vnd.github.text-match+json',
    'Authorization': f'Bearer {GITHUB_TOKEN}',
    'X-GitHub-Api-Version': '2022-11-28'
}

def robust_request(url, headers=None, max_retries=3, timeout=TIMEOUT, verify_ssl=True):
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True
    )
    headers = headers or {}

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    for attempt in range(max_retries + 1):
        try:
            response = session.get(
                url,
                timeout=timeout,
                verify=verify_ssl,
                headers=headers
            )
            response.raise_for_status()
            return response

        except requests.exceptions.SSLError as e:
            logging.error(f"SSL error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries:
                logging.warning("Attempting retry with SSL verification disabled...")
                try:
                    return session.get(url, timeout=timeout, verify=False, headers=headers)
                except Exception as final_e:
                    logging.error(f"Failed after disabling SSL verification: {final_e}")
                    return None
            time.sleep(2 ** attempt)

        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries:
                return None
            time.sleep(2 ** attempt)
    return None

def search_sourcegraph(api):
    items = []
    
    def search(keywords):
        # url = SOURCEGRAPH_SEARCH_URL.format(keywords=keywords)
        url = SOURCEGRAPH_SEARCH_URL.format(keywords=keywords, max_count=MAX_COUNT)
        logging.info(f"Sourcegraph search URL: {url}")
        response = requests.get(url, headers=SOURCEGRAPH_HEADERS, stream=True, timeout=10)
        if response.status_code == 200:
            lines = response.text.split('\n')
            idx = 0
            while idx < len(lines):
                if lines[idx] == "event: matches":
                    item_str = lines[idx+1][6:]
                    items.extend(json.loads(item_str))
                    idx += 1
                idx += 1
        else:
            logging.error(f"Error: {response.status_code}, {response.text}")
                    
    full_names = {f"{api}("}
    full_names.update(f"{api.replace(lib, alias)}(" for alias in ALIAS.get(lib, []))
    keywords = " OR ".join(full_names)
    search(keywords)
    
    parts: List[str] = api.split(".")
    if len(parts) >= 2:
        if parts[-1][0].isupper():
            keywords = [".".join(parts[:-1]), f"{parts[-1]}("]
        elif parts[-2][0].isupper():
            if len(parts) > 2:
                keywords = [".".join(parts[:-2]), f"{parts[-2]}", f".{parts[-1]}("]
            else:
                keywords = [f"{parts[-2]}", f".{parts[-1]}("]
        else:
            keywords = [".".join(parts[:-1]), f".{parts[-1]}("]
        keywords=" ".join(keywords)
    search(keywords)

    items = [item for item in items if "repoStars" in item]
    items.sort(key=lambda x: x["repoStars"], reverse=True)
    logging.info(f"Sourcegraph search completed, {len(items)} valid results")
    return items


def search_github(api):
    github_items = []
    fetched_keys = set()

    def search(keywords):
        logging.info(f"GitHub search keyword: {keyword}")
        encoded_keyword = urllib.parse.quote(keyword)

        first_page_url = f"{GITHUB_SEARCH_BASE_URL}?q={encoded_keyword}+in:file+language:python&per_page=100&page=1&sort=stars&order=desc"
        first_response = robust_request(first_page_url, headers=GITHUB_HEADERS)
        if not first_response:
            logging.error(f"GitHub first page search failed for keyword: {keyword}, skipping")
            return
        first_data = first_response.json()
        if "total_count" not in first_data:
            logging.error(f"GitHub response abnormal: {first_data}")
            return
        total_count = first_data["total_count"]
        logging.info(f"GitHub total results for keyword {keyword}: {total_count}")
        if total_count == 0:
            return

        max_page = min(GITHUB_MAX_PAGE, (total_count // 100) + 1)
        for page in range(1, max_page + 1):
            search_url = f"{GITHUB_SEARCH_BASE_URL}?q={encoded_keyword}+in:file+language:python&per_page=100&page={page}&sort=stars&order=desc"
            response = robust_request(search_url, headers=GITHUB_HEADERS)
            if not response:
                logging.error(f"GitHub page {page} search failed, skipping")
                time.sleep(10)
                continue
            data = response.json()
            if "items" not in data:
                logging.error(f"GitHub page {page} has no results: {data}")
                time.sleep(10)
                continue
            for item in data["items"]:
                try:
                    repo_name = item["repository"]["full_name"]
                    # commit = item["url"].split("ref=")[1]
                    # file_path = item["path"]
                    file_path = item["path"]
                    refs = urllib.parse.parse_qs(urllib.parse.urlparse(item["url"]).query)
                    commit = refs.get("ref", [None])[0]
                    if not commit:
                        # GitHub code-search URLs may omit ref; the HTML URL includes it.
                        blob_path = urllib.parse.unquote(
                            urllib.parse.urlparse(item["html_url"]).path
                        ).split("/blob/", 1)[1]
                        suffix = "/" + file_path
                        if not blob_path.endswith(suffix):
                            raise ValueError("Cannot extract GitHub reference")
                        commit = blob_path[:-len(suffix)]
                    unique_key = f"{repo_name}/{commit}/{file_path}"
                    if unique_key in fetched_keys:
                        continue
                    github_item = {
                        "repository": f"github.com/{repo_name}",
                        "commit": commit,
                        "path": file_path,
                        "source": "github"
                    }
                    github_items.append(github_item)
                    fetched_keys.add(unique_key)
                # except KeyError as e:
                except (KeyError, IndexError, ValueError) as e:
                    logging.error(f"Missing field in GitHub result: {e}, raw data: {item}")
                    continue
            time.sleep(10)
    
    full_names = {f"{api}("}  
    full_names.update(f"{api.replace(prefix, alias)}("  for prefix, alias_list in ALIAS.items() for alias in alias_list)
    for keyword in list(full_names):
        search(keyword)

    parts = api.split(".")
    if len(parts) >= 2:
        if parts[-1][0].isupper():
            keyword = [".".join(parts[:-1]), f"{parts[-1]}("]
        elif parts[-2][0].isupper():
            if len(parts) > 2:
                keyword = [".".join(parts[:-2]), f"{parts[-2]}", f".{parts[-1]}("]
            else:
                keyword = [f"{parts[-2]}", f".{parts[-1]}("]
        else:
            keyword = [".".join(parts[:-1]), f".{parts[-1]}("]
        keyword = " ".join(keyword)
        search(keyword)

    logging.info(f"GitHub search completed, {len(github_items)} valid results")
    return github_items

def merge_and_deduplicate(sg_items, github_items):
    merged = []
    seen_keys = set()

    for item in sg_items:
        unique_key = f"{item['repository'].replace('github.com/', '')}/{item['commit']}/{item['path']}"
        if unique_key not in seen_keys:
            seen_keys.add(unique_key)
            merged.append(item)

    for item in github_items:
        unique_key = f"{item['repository'].replace('github.com/', '')}/{item['commit']}/{item['path']}"
        if unique_key not in seen_keys:
            seen_keys.add(unique_key)
            merged.append(item)

    merged = merged[:MAX_COUNT] if len(merged) > MAX_COUNT else merged
    logging.info(f"Merge and deduplication completed, {len(merged)} final results")
    return merged

def download_source_files(items: List[dict], output_dir: str):
    """Download source code files"""
    fetched_files = set()
    source_dir = Path(output_dir) / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(items, desc="Downloading sources"):
        repo = item["repository"].replace("github.com/", "")
        commit = item["commit"]
        file_path = item["path"]
        unique_key = f"{repo}/{commit}/{file_path}"
        if unique_key in fetched_files:
            continue
        fetched_files.add(unique_key)

        repo_safe = repo.replace("/", "#")
        save_path = source_dir / f"{repo_safe}-{commit}" / file_path
        if save_path.exists():
            continue
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Download file
        raw_url = GITHUB_RAW_URL.format(repo=repo, commit=commit, file=file_path)
        logging.info(f"Downloading file: {raw_url}")
        response = robust_request(raw_url, headers={"Authorization": f"token {GITHUB_TOKEN}"})

        if response:
            try:
                with save_path.open("w", encoding="utf-8") as f:
                    f.write(response.text)
            except Exception as e:
                logging.error(f"Failed to save file {save_path}: {e}")
        else:
            logging.error(f"Failed to download file {raw_url}")

        time.sleep(1)  # Control request frequency for GitHub raw files


if __name__ == '__main__':
    for lib in LIBs:
        mappings_file = Path(f"data/mappings/deprecated-mappings-{lib}.json")
        output_dir = Path(f"data/searching-results/{lib}/")
        output_dir.mkdir(parents=True, exist_ok=True)
        init_log(f"{output_dir}/search.log")  # Initialize logger
        logging.info(f"Starting processing for library: {lib}")

        if not mappings_file.exists():
            logging.error(f"Mappings file {mappings_file} not found, skipping library")
            continue
        with mappings_file.open("r") as f:
            apis = set([replacement for _, replacement in json.load(f).items()])
        apis = list(sorted(apis))
        logging.info(f"Loaded {len(apis)} APIs to search")

        for api in tqdm(apis, desc=f"Processing {lib} APIs"):
            result_file = output_dir / f"{api}.json"
            # if result_file.exists():
                # logging.info(f"Result for API {api} already exists, skipping")
                # continue
            if result_file.exists():
                with result_file.open("r", encoding="utf-8") as f:
                    cached_items = json.load(f)
                if len(cached_items) >= MAX_COUNT:
                    logging.info(f"Reusing cached results for API {api}")
                    download_source_files(cached_items[:MAX_COUNT], str(output_dir))
                    continue
                logging.info(f"Refreshing results for API {api}: fewer than {MAX_COUNT} candidates")
            logging.info(f"Starting processing for API: {api}")

            sg_items = search_sourcegraph(api)
            github_items = search_github(api)

            final_items = merge_and_deduplicate(sg_items, github_items)

            with result_file.open("w", encoding="utf-8") as f:
                json.dump(final_items, f, indent=4)
            logging.info(f"Results for API {api} saved to {result_file}")

            # Download source files
            if final_items:
                download_source_files(final_items, str(output_dir))

        logging.info(f"Processing completed for library: {lib}\n{'='*50}")