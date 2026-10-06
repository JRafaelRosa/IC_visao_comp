import kagglehub

# Download latest version
path = kagglehub.dataset_download("simuletic/cctv-pedestrian-1k-person-attribute-dataset")

print("Path to dataset files:", path)