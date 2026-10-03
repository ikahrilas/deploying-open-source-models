from huggingface_hub import scan_cache_dir

cache_info = scan_cache_dir()
print(f"Total cache size: {cache_info.size_on_disk_str}")
for repo in cache_info.repos:
    print(f"  {repo.repo_id}: {repo.size_on_disk_str}")