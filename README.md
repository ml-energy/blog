# ML.ENERGY Research and Tech Blog

[![Blog deploy](https://github.com/ml-energy/blog/actions/workflows/deploy_homepage.yaml/badge.svg)](https://github.com/ml-energy/blog/actions/workflows/deploy_homepage.yaml)

## Structure

- `docs/posts/`: Blog post markdown files.
- `analysis/`: Jupyter Notebooks and Python scripts that reproduce numbers, analyses, and figures in blog posts. One file per post, named to match the post file.

## Install dependencies

```sh
pip install -r requirements.txt
```

## Build

```sh
mkdocs build
```

## Live preview

```sh
mkdocs serve 
```
